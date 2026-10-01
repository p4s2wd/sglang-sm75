# sglang SM75 release artifacts

Release artifacts for the sub-90 (SM75/SM80) DeepSeek-V4-Flash path, kept
outside the sglang fork so the fork stays a source tree and the binaries stay
versioned.

- `release-sm75-main2/` — the current release (`sm75main2`): the pure-python
  wheel, the patch it is built from, the docs, checksums.
- The code it is built from lives in the fork
  [p4s2wd/sglang-sm75](https://github.com/p4s2wd/sglang-sm75) on branch
  `sm75-dsv4-flash-main`, tagged `v0.2.0-sm75main2`.

## Releases

| tag | base | notes |
|---|---|---|
| `v0.2.0-sm75main2` | upstream main `98fce73d5b` | **fixes an illegal memory access that crashes sm75main1 on the stock path**; plus decode work and the F3 layer split as the launcher default (+7.6% prefill) |
| `v0.2.0-sm75main1` | upstream main `98fce73d5b` | opt3+opt4 rebased; prefill +9-11%, long-context decode +10% vs the 0.5.19 base. **Superseded — it carries the IMA crash** |
| `v0.1.0-sm75opt1..4` | 0.5.19.dev332 | shipped as local wheels and patches; the tags on the fork point at the base commit, not at the code |

## ⚠ Do not deploy sm75main1 or older

`dsv4/topk.py`'s paged top-k transform took `page_table_width` and never used
it, so a stale token produced a page id past the end of the row (observed:
12673630) and the sparse-attention gather used it as a KV address. The device
faults within the first few greedy prompts on the default DeepSeek-V4-Flash
path. `v0.2.0-sm75main2` bounds it; out-of-range entries go 4 -> 0.

## Verify a download

```bash
cd release-sm75-main2 && sha256sum -c SHA256SUMS && sha256sum -c MANIFEST.sha256
```

The wheel is byte-reproducible from the tag: `build-wheel.sh` in that directory
rebuilds it from a checkout of the tagged commit and produces the same sha256.
