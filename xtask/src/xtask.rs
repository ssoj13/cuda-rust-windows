// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generator of the committed `cuda-bindings/generated/` tree.
//!
//! `cuda-bindings` ships its bindgen output instead of running bindgen in
//! `build.rs`, so consumers build without a CUDA toolkit (the driver is still
//! loaded per symbol at run time). This crate owns the generator: bindgen over
//! `cuda.h` / `curand.h`, the dynamic-loader API structs and their `extern "C"`
//! shims, and the cfgs recorded in `generated/meta.json`. The CLI is
//! `cargo xtask bindings`; the byte-equality test lives in `tests/`.
//!
//! Why an xtask and not `build.rs`: generation needs libclang and a toolkit, which
//! consumers should not need; it runs when a developer bumps the toolkit, and the
//! result is reviewed and committed. Why per-target sets: bindgen output differs
//! per platform (MSVC `c_int` enums vs `c_uint` elsewhere, layout asserts).
//!
//! The supported-target table and toolkit discovery are shared with
//! `cuda-bindings/build.rs` (one copy, `cuda-bindings/shared.rs`).

// `fs`, `env` and the `Path` types come from the shared include below.
use std::{collections::BTreeSet, error::Error};

use bindgen::CodegenConfig;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use syn::{
    Expr, Fields, File, GenericArgument, Item, ItemStruct, PathArguments, ReturnType, Type,
    TypeBareFn,
};

include!("../../cuda-bindings/shared.rs");

/// Result alias; every failure here is reported to the developer as text.
pub type Res<T> = Result<T, Box<dyn Error>>;

/// `generated/meta.json` schema version, bumped when the layout changes.
const META_SCHEMA: u64 = 1;

/// One generated dynamic-loader API: bindgen over `header_contents` plus the
/// shims for the functions matching `function_pattern`.
struct ApiSpec {
    virtual_header: &'static str,
    header_contents: &'static str,
    generated_api: &'static str,
    generated_shims: &'static str,
    library_name: &'static str,
    api_type: &'static str,
    loader_fn: &'static str,
    /// Returned when the library itself could not be loaded.
    not_loaded_expr: &'static str,
    /// Returned when the library loaded but this symbol is absent (older
    /// driver). Matches cuGetProcAddress's own convention.
    missing_symbol_expr: &'static str,
    function_pattern: &'static str,
}

const API_SPECS: &[ApiSpec] = &[
    ApiSpec {
        virtual_header: "cuda_driver_wrapper.h",
        header_contents: "#include <cuda.h>\n",
        generated_api: "cuda_driver_api.rs",
        generated_shims: "cuda_driver_shims.rs",
        library_name: "CudaDriverApi",
        api_type: "CudaDriverApi",
        loader_fn: "cuda_driver",
        not_loaded_expr: "cudaError_enum_CUDA_ERROR_NOT_INITIALIZED",
        missing_symbol_expr: "cudaError_enum_CUDA_ERROR_NOT_FOUND",
        function_pattern: "^cu.*",
    },
    ApiSpec {
        virtual_header: "curand_wrapper.h",
        header_contents: "#include <curand.h>\n",
        generated_api: "curand_api.rs",
        generated_shims: "curand_shims.rs",
        library_name: "CurandApi",
        api_type: "CurandApi",
        loader_fn: "curand_api",
        // curand has no NOT_FOUND analogue; NOT_INITIALIZED covers both.
        not_loaded_expr: "curandStatus_CURAND_STATUS_NOT_INITIALIZED",
        missing_symbol_expr: "curandStatus_CURAND_STATUS_NOT_INITIALIZED",
        function_pattern: "^curand.*",
    },
];

/// Everything generated for one target: the files of `generated/<key>/` and the
/// target's `meta.json` entry.
pub struct Output {
    pub key: &'static str,
    pub files: Vec<(&'static str, String)>,
    pub meta: Value,
}

/// `cuda-bindings/` in this checkout.
pub fn bindings_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../cuda-bindings")
}

/// The committed tree, `cuda-bindings/generated/`.
pub fn generated_dir() -> PathBuf {
    bindings_dir().join("generated")
}

/// Every supported target key, in table order.
pub fn all_keys() -> Vec<&'static str> {
    TARGETS.iter().map(|t| t.key).collect()
}

/// The supported target this process runs on, if any.
pub fn host_key() -> Option<&'static str> {
    let env = if cfg!(target_env = "msvc") {
        "msvc"
    } else if cfg!(target_env = "gnu") {
        "gnu"
    } else {
        ""
    };
    find_target(std::env::consts::OS, std::env::consts::ARCH, env).map(|t| t.key)
}

/// Generates the bindings of `key` from a toolkit (`toolkit`, else the
/// environment / default locations). The target is passed to clang as
/// `--target=<key>`; `clang_args` supplies what a cross build needs (for
/// example `--sysroot=<dir>` with the target's libc headers).
pub fn generate(key: &str, toolkit: Option<&Path>, clang_args: &[String]) -> Res<Output> {
    let target = TARGETS
        .iter()
        .find(|t| t.key == key)
        .ok_or_else(|| format!("unsupported target {key}; supported: {}", all_keys().join(", ")))?;
    let tk = discover_toolkit(toolkit, target.arch, target.os)?.ok_or_else(|| {
        format!(
            "no CUDA 13.0+ toolkit found: pass --toolkit, or set {}",
            TOOLKIT_ENV_VARS.join(" / ")
        )
    })?;
    let cuda_h = fs::read_to_string(tk.include_dir.join("cuda.h"))?;
    let curand_h = fs::read(tk.include_dir.join("curand.h"))
        .map_err(|e| format!("{}/curand.h: {e}", tk.include_dir.display()))?;

    let builder = || {
        bindgen::builder()
            .clang_arg(format!("-I{}", tk.include_dir.display()))
            .clang_arg(format!("--target={key}"))
            .clang_args(clang_args)
    };

    let mut files = Vec::new();
    let types = builder()
        .header(bindings_dir().join("wrapper.h").to_string_lossy())
        .blocklist_function(".*")
        .generate()?
        .to_string();
    let mut cfgs = BTreeSet::new();
    if cu_mem_location_uses_anon_union(&types)? {
        cfgs.insert(BINDINGS_CFGS[0]);
    }
    for (cfg, symbol) in [
        (BINDINGS_CFGS[1], "cuEventElapsedTime_v2"),
        (BINDINGS_CFGS[2], "cuLaunchHostFunc_v2"),
        (BINDINGS_CFGS[3], "cuMulticastCreate"),
    ] {
        if cuda_h.contains(symbol) {
            cfgs.insert(cfg);
        }
    }
    files.push(("types.rs", types));

    for spec in API_SPECS {
        let api = builder()
            .header_contents(spec.virtual_header, spec.header_contents)
            .dynamic_library_name(spec.library_name)
            .dynamic_link_require_all(false)
            .allowlist_function(spec.function_pattern)
            .with_codegen_config(CodegenConfig::FUNCTIONS)
            .generate()?
            .to_string();
        let shims = generate_shims(&api, spec)?;
        files.push((spec.generated_api, api));
        files.push((spec.generated_shims, shims));
    }

    let meta = json!({
        "cuda_version": tk.version,
        "cuda_h_sha256": hex(&Sha256::digest(cuda_h.as_bytes())),
        "curand_h_sha256": hex(&Sha256::digest(&curand_h)),
        "cfgs": cfgs,
    });
    Ok(Output { key: target.key, files, meta })
}

/// Writes `out` under `root/<key>/` and merges its entry into `root/meta.json`
/// (other targets' entries are kept, so each platform can be regenerated on
/// its own machine).
pub fn write(out: &Output, root: &Path) -> Res<()> {
    let dir = root.join(out.key);
    fs::create_dir_all(&dir)?;
    for (name, text) in &out.files {
        fs::write(dir.join(name), text)?;
    }
    let meta_path = root.join("meta.json");
    let mut meta = match fs::read_to_string(&meta_path) {
        Ok(text) => serde_json::from_str::<Value>(&text)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => json!({}),
        Err(e) => return Err(e.into()),
    };
    let root_obj = meta.as_object_mut().ok_or("meta.json is not an object")?;
    root_obj.insert("schema".into(), META_SCHEMA.into());
    let targets = root_obj
        .entry("targets")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or("meta.json: `targets` is not an object")?;
    targets.insert(out.key.into(), out.meta.clone());
    fs::write(&meta_path, serde_json::to_string_pretty(&meta)? + "\n")?;
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Whether bindgen wrapped `CUmemLocation_st::id` in an anonymous union
/// (CUDA 13.2+) or left it a plain `int` (13.0 / 13.1); `set_mem_location_id`
/// in `lib.rs` picks its access path from this.
fn cu_mem_location_uses_anon_union(generated_source: &str) -> Res<bool> {
    let file = syn::parse_file(generated_source)?;
    Ok(file.items.iter().any(|item| match item {
        Item::Struct(s) if s.ident == "CUmemLocation_st" => {
            let Fields::Named(fields) = &s.fields else {
                return false;
            };
            fields
                .named
                .iter()
                .any(|f| f.ident.as_ref().is_some_and(|i| i == "__bindgen_anon_1"))
        }
        _ => false,
    }))
}

fn generate_shims(generated_source: &str, spec: &ApiSpec) -> Res<String> {
    let file = syn::parse_file(generated_source)?;
    let item_struct = find_api_struct(&file, spec.api_type).ok_or_else(|| {
        format!(
            "no generated struct found for {} in {}",
            spec.api_type, spec.generated_api
        )
    })?;

    let loader_fn = format_ident!("{}", spec.loader_fn);
    let not_loaded_expr = syn::parse_str::<Expr>(spec.not_loaded_expr)?;
    let missing_symbol_expr = syn::parse_str::<Expr>(spec.missing_symbol_expr)?;
    let shims = generate_field_shims(item_struct, &loader_fn, &not_loaded_expr, &missing_symbol_expr);
    let shim_file = syn::parse2::<File>(quote!(#shims))?;
    Ok(prettyplease::unparse(&shim_file))
}

fn find_api_struct<'a>(file: &'a File, api_type: &str) -> Option<&'a ItemStruct> {
    file.items.iter().find_map(|item| match item {
        Item::Struct(item_struct) if item_struct.ident == api_type => Some(item_struct),
        _ => None,
    })
}

fn generate_field_shims(
    item_struct: &ItemStruct,
    loader_fn: &proc_macro2::Ident,
    not_loaded_expr: &Expr,
    missing_symbol_expr: &Expr,
) -> TokenStream {
    let Fields::Named(fields) = &item_struct.fields else {
        return TokenStream::new();
    };

    let shim_fns = fields.named.iter().filter_map(|field| {
        let field_name = field.ident.as_ref()?;
        let bare_fn = result_bare_fn(&field.ty)?;
        let args = bare_fn
            .inputs
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                let name = arg
                    .name
                    .as_ref()
                    .map(|(ident, _)| ident.clone())
                    .unwrap_or_else(|| format_ident!("arg_{index}"));
                let ty = &arg.ty;
                (name.clone(), quote!(#name: #ty))
            })
            .collect::<Vec<_>>();
        let arg_names = args.iter().map(|(name, _)| name).collect::<Vec<_>>();
        let arg_defs = args.iter().map(|(_, def)| def).collect::<Vec<_>>();
        let ret = match &bare_fn.output {
            ReturnType::Default => quote!(),
            ReturnType::Type(_, ty) => quote!(-> #ty),
        };

        // `extern "C"` so a wrapper item coerces to a C function pointer,
        // matching cuda-oxide's original bindings surface (deliberately not
        // "C-unwind": that is a different fn-pointer type and breaks the
        // coercion). Unwinding out of extern "C" aborts, so these bodies
        // must be panic-free: both failure paths return the API's own
        // error codes instead.
        Some(quote! {
            #[allow(non_snake_case)]
            #[allow(clippy::missing_safety_doc)]
            #[allow(clippy::too_many_arguments)]
            #[allow(clippy::missing_inline_in_public_items)]
            #[inline]
            pub unsafe extern "C" fn #field_name(#(#arg_defs),*) #ret {
                match #loader_fn() {
                    Ok(api) => match &api.#field_name {
                        Ok(loaded_fn) => unsafe { (*loaded_fn)(#(#arg_names),*) },
                        Err(_) => #missing_symbol_expr,
                    },
                    Err(_) => #not_loaded_expr,
                }
            }
        })
    });

    quote! {
        #(#shim_fns)*
    }
}

fn result_bare_fn(ty: &Type) -> Option<&TypeBareFn> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Result" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let Some(GenericArgument::Type(Type::BareFn(bare_fn))) = args.args.first() else {
        return None;
    };
    Some(bare_fn)
}
