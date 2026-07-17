#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

# The checked-in Cargo config contains paths from the original development
# machine. Prefer explicit caller settings, otherwise discover local tools.
if [[ -z "${CMAKE:-}" ]]; then
    CMAKE="$(command -v cmake || true)"
    if [[ -n "$CMAKE" ]]; then
        export CMAKE
    fi
fi

if [[ -z "${LIBCLANG_PATH:-}" ]]; then
    for dir in /usr/lib /usr/lib64; do
        if compgen -G "$dir/libclang.so*" >/dev/null; then
            export LIBCLANG_PATH="$dir"
            break
        fi
    done
fi

if [[ -z "${BINDGEN_EXTRA_CLANG_ARGS:-}" ]] && command -v clang >/dev/null; then
    clang_resource_dir="$(clang -print-resource-dir)"
    export BINDGEN_EXTRA_CLANG_ARGS="-I${clang_resource_dir}/include"
fi

exec cargo run -p keyb-app --bin keyb -- "$@"
