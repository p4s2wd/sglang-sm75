# sglang SM75 release artifacts

Release artifacts for the sub-90 (SM75/SM80) DeepSeek-V4-Flash path, kept
outside the sglang fork so the fork stays a source tree and the binaries stay
versioned.

- `release-sm75-main3/` — the current release (`sm75main3`): the pure-python
  wheel, the patches it is built from, the docs, checksums.
- The code it is built from lives in the fork
  [p4s2wd/sglang-sm75](https://github.com/p4s2wd/sglang-sm75) on branch
  `sm75-dsv4-flash-main`, tagged `v0.2.1-sm75main3`.
- `release-sm75-main2/` is on the same branch's history, one release behind.

## Releases

| tag | base | notes |
|---|---|---|
| `v0.2.1-sm75main3` | upstream main `98fce73d5b` | **the 256K context becomes real** (SWA bytes were charged to every layer, so the KV pool was 162,304 tokens instead of the 262,144 declared; now 264,192 on the reference box); **tool calls survive model drift** (a missing `string` attribute, a closer missing its `｜DSML｜` prefix, or a tool name with the wrong case no longer costs the whole call); long-context sparse top-K slabbed across SMs, 1.7x at 8K row width to 25.6x at 262K, byte-identical output |
| `v0.2.0-sm75main2` | upstream main `98fce73d5b` | **fixed the illegal memory access that crashes sm75main1 on the stock path**; decode work; the F3 layer split as the launcher default (+7.6% prefill). **Superseded by sm75main3** — it runs, but its KV pool is 162,304 tokens, so `--context-length 262144` there is fiction, and a tool call with any format drift is dropped |
| `v0.2.0-sm75main1` | upstream main `98fce73d5b` | opt3+opt4 rebased; prefill +9-11%, long-context decode +10% vs the 0.5.19 base. **Superseded — it carries the IMA crash** |
| `v0.1.0-sm75opt1..4` | 0.5.19.dev332 | shipped as local wheels and patches; the tags on the fork point at the base commit, not at the code |

## ⚠ Do not deploy sm75main1 or older

`dsv4/topk.py`'s paged top-k transform took `page_table_width` and never used
it, so a stale token produced a page id past the end of the row (observed:
12673630) and the sparse-attention gather used it as a KV address. The device
faults within the first few greedy prompts on the default DeepSeek-V4-Flash
path. `v0.2.0-sm75main2` bounds it; out-of-range entries go 4 -> 0.

## ⚠ sm75main3 changes the memory shape, not just the code

The SWA accounting fix turns 0.18 GB of PP0 headroom into KV pool on purpose:
PP0 ends at 0.46 GB and **PP3 at 0.64 GB becomes the binding rank**. Re-check
PP3 before raising `--mem-fraction-static`, adding graph buckets, or raising
`--chunked-prefill-size` above the shipped 256.

## Verify a download

```bash
cd release-sm75-main3 && sha256sum -c SHA256SUMS && sha256sum -c MANIFEST.sha256
```

The wheel is byte-reproducible from the tag: `build-wheel.sh` in that directory
rebuilds it from a checkout of the tagged commit and produces the same sha256.
Checked for this release by building `a588149ab0` twice. Both patches were also
applied from their stated base commits and diffed against the released tree:
`sm75-optimizations.patch` onto upstream `98fce73d5b`, and
`sm75-main3-series.patch` onto `sm75main2`'s `331faaeaf7`; both reproduce it
exactly.
