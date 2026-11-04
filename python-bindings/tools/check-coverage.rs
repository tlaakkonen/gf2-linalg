use std::{collections::HashSet, path::PathBuf};
use std::fs::File;
use std::io::BufReader;
use std::process::{Command, ExitCode};
use pyo3_introspection::model::Module;
use rustdoc_types::{Crate, Item as RItem, ItemEnum};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Item {
    Struct(String),
    Method(String, String),
    Function(String)
}

impl std::fmt::Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::Struct(name) => write!(f, "struct {}", name),
            Item::Method(class, name) => write!(f, "{}::{}", class, name),
            Item::Function(name) => write!(f, "fn {}", name)
        }
    }
}

fn inspect_rustdoc_item(krate: &Crate, item: &RItem, parent: Option<&str>, out: &mut HashSet<Item>) {
    match &item.inner {
        ItemEnum::Module(module) => {
            for child in &module.items {
                inspect_rustdoc_item(krate, &krate.index[child], None, out);
            }
        },
        ItemEnum::Use(u) => if let Some(id) = &u.id {
            inspect_rustdoc_item(krate, &krate.index[id], None, out);
        },
        ItemEnum::Struct(st) => {
            out.insert(Item::Struct(item.name.clone().unwrap()));
            for child in &st.impls {
                inspect_rustdoc_item(krate, &krate.index[child], Some(item.name.as_ref().unwrap()), out);
            }
        },
        ItemEnum::Impl(imp) => {
            if imp.trait_.is_some() { return }
            for child in &imp.items {
                inspect_rustdoc_item(krate, &krate.index[child], parent, out);
            }
        },
        ItemEnum::Function(_) | ItemEnum::AssocConst { .. } => {
            if let Some(parent) = parent {
                out.insert(Item::Method(parent.to_string(), item.name.clone().unwrap()));
            } else {
                out.insert(Item::Function(item.name.clone().unwrap()));
            }
        },
        ItemEnum::Trait(_) => return,
        _ => unimplemented!("{:?}", item)
    }
}

fn inspect_pyo3_module(module: &Module, items: &mut HashSet<Item>) {
    for submodule in &module.modules {
        inspect_pyo3_module(submodule, items);
    }

    for class in &module.classes {
        items.insert(Item::Struct(class.name.clone()));

        for function in &class.methods {
            items.insert(Item::Method(class.name.clone(), function.name.clone()));
        }

        for attr in &class.attributes {
            items.insert(Item::Method(class.name.clone(), attr.name.clone()));
        }
    }

    for function in &module.functions {
        items.insert(Item::Function(function.name.clone()));
    }
}

fn main() -> ExitCode {
    let filter = std::env::args().nth(1).map(|filter| regex::Regex::new(&filter).unwrap());

    let rustdoc_status = Command::new("cargo")
        .arg("+nightly")
        .arg("doc")
        .arg("-p").arg("gf2-linalg")
        .arg("--features").arg("rand")
        .arg("--no-deps")
        .arg("-Z").arg("unstable-options")
        .arg("--output-format").arg("json")
        .status().unwrap();
    if !rustdoc_status.success() { panic!() }

    let cargo_status = Command::new("cargo")
        .arg("build")
        .arg("-p").arg("python-bindings")
        .status().unwrap();
    if !cargo_status.success() { panic!() }

    let json_filename = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join( "../../target/doc/gf2_linalg.json");
    let file = File::open(json_filename).unwrap();
    let reader = BufReader::new(file);
    let krate: Crate = serde_json::from_reader(reader).unwrap();
    let mut out = HashSet::new();
    inspect_rustdoc_item(&krate, &krate.index[&krate.root], None, &mut out);
    let mut rust_items = out.into_iter().collect::<Vec<_>>();
    rust_items.sort();

    let cdylib_filename = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join( "../../target/debug/libgf2_linalg.so");
    let pyo3_module = pyo3_introspection::introspect_cdylib(cdylib_filename, "gf2_linalg").unwrap();
    let mut python_items = HashSet::new();
    inspect_pyo3_module(&pyo3_module, &mut python_items);
    let mut python_items = python_items.into_iter().collect::<Vec<_>>();
    python_items.sort();

    let skips = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join( "coverage-skips.txt");
    let skips = if skips.exists() {
        let skips_text = std::fs::read_to_string(skips).unwrap();
        regex::RegexSet::new(
            skips_text.split('\n').filter(|s| !(s.starts_with('#') || s.is_empty()))
        ).unwrap()
    } else {
        regex::RegexSet::new([] as [&str;0]).unwrap()
    };

    let mut found = 0;
    let mut missing = 0;
    let mut skipped = 0;
    for item in rust_items {
        if skips.is_match(&format!("{}", item)) {
            skipped += 1;
            continue
        }

        if python_items.contains(&item) {
            found += 1;
        } else {
            if filter.as_ref().map(|f| f.is_match(&format!("{}", item))).unwrap_or(true) {
                println!("missing: {}", item);
            }
            missing += 1;
        }
    }
    println!("found = {}, missing = {}, skipped = {}", found, missing, skipped);

    if missing > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}