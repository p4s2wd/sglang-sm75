#!/usr/bin/env bash
# Rebuild the sm75 wheel from a checkout of the rebased tree.
#
#   CHECKOUT=/path/to/sglang ./build-wheel.sh
#
# Unlike release-sm75-opt4, this does not repack a base wheel: the opt3/opt4
# wheels were overlays on a 0.5.19 base, and this release is a different base
# (current main), so there is nothing to overlay onto. It builds the checkout
# directly and the result is a pure-python wheel, same tag as every earlier
# sm75 release.
#
# Two things have to be handled explicitly, both upstream packaging quirks:
#
#   SGLANG_BUILD_RUST_EXTS=none
#       sglang discovers Rust extensions from ../rust and compiles them by
#       default, which takes ~30 min and turns the artifact into a
#       cp312-linux wheel. sgl-kernel and sgl-router ship separately, so the
#       pure-python wheel is what every sm75 release has been.
#
#   python/sglang/multimodal_gen/.claude/skills
#       A symlink to ../.agents/skills. setuptools' file finder lists it as a
#       file and then refuses to copy it ("doesn't exist or not a regular
#       file"), so the build dies late. Replacing it with a real directory
#       gets the build past it. It touches only the worktree, never the
#       checkout.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="${CHECKOUT:?set CHECKOUT to the sglang checkout to build}"
PYTHON="${PYTHON:-python3}"
VERSION="${VERSION:-0.5.21.dev797+ge0f76063c.sm75main1}"
OUT_DIR="${OUT_DIR:-$HERE}"

command -v cargo >/dev/null 2>&1 && echo "note: cargo present but SGLANG_BUILD_RUST_EXTS=none disables the Rust build"

BUILD_SRC="$(mktemp -d)"
trap 'git -C "$CHECKOUT" worktree remove --force "$BUILD_SRC" >/dev/null 2>&1 || rm -rf "$BUILD_SRC"' EXIT

# A detached worktree, not `git archive`: setuptools-scm's file finder reads
# git metadata to decide which files are package data, and without it the wheel
# silently loses dotfiles and the agent docs (56 entries: .clang-format,
# .flake8, multimodal_gen/.agents/**). A worktree is clean, so the build still
# cannot pick up local edits or build artefacts.
git -C "$CHECKOUT" worktree add --detach "$BUILD_SRC" HEAD >/dev/null

# Dereference the symlink setuptools cannot copy (see the note above).
rm "$BUILD_SRC/python/sglang/multimodal_gen/.claude/skills"
cp -r "$BUILD_SRC/python/sglang/multimodal_gen/.agents/skills" \
      "$BUILD_SRC/python/sglang/multimodal_gen/.claude/skills"

echo "building $VERSION from $CHECKOUT ($(git -C "$CHECKOUT" rev-parse --short HEAD))"
# SOURCE_DATE_EPOCH makes the zip byte-reproducible: without it every rebuild
# differs in the entry timestamps and therefore in the sha256.
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$CHECKOUT" log -1 --format=%ct HEAD)}" 
SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION" \
  "$PYTHON" -m build --wheel --no-isolation --outdir "$OUT_DIR" \
  "$BUILD_SRC/python"

WHEEL="$OUT_DIR/sglang-$VERSION-py3-none-any.whl"
test -f "$WHEEL"

# Normalise the zip entry timestamps. `wheel` 0.48 does not honour
# SOURCE_DATE_EPOCH, so without this every rebuild is a different sha256 even
# though every entry is byte-identical. After this, two builds of the same
# commit produce the same file and the checksum below is worth something.
"$PYTHON" - "$WHEEL" "$SOURCE_DATE_EPOCH" <<'NORMALISE'
import sys, time, zipfile

wheel, epoch = sys.argv[1], int(sys.argv[2])
stamp = time.gmtime(epoch)[:6]
src = zipfile.ZipFile(wheel)
with zipfile.ZipFile(wheel + ".tmp", "w", zipfile.ZIP_DEFLATED) as dst:
    for info in sorted(src.infolist(), key=lambda i: i.filename):
        out = zipfile.ZipInfo(info.filename, date_time=stamp)
        out.external_attr = info.external_attr
        out.create_system = info.create_system
        out.compress_type = info.compress_type
        dst.writestr(out, src.read(info.filename))
src.close()
import os
os.replace(wheel + ".tmp", wheel)
NORMALISE

echo
echo "$WHEEL"
sha256sum "$WHEEL"
