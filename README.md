# sglang SM75 release artifacts

Release artifacts for the sub-90 (SM75/SM80) DeepSeek-V4-Flash path, kept
outside the sglang fork so the fork stays a source tree and the binaries stay
versioned.

- `release-sm75-main/` — the current release (`sm75main1`): the pure-python
  wheel, the patch it is built from, the docs, checksums.
- The code it is built from lives in the fork
  [p4s2wd/sglang-sm75](https://github.com/p4s2wd/sglang-sm75) on branch
  `sm75-dsv4-flash-main`, tagged `v0.2.0-sm75main1`.

## Releases

| tag | base | notes |
|---|---|---|
| `v0.2.0-sm75main1` | upstream main `98fce73d5b` | opt3+opt4 rebased; prefill +9-11%, long-context decode +10% vs the 0.5.19 base |
| `v0.1.0-sm75opt1..4` | 0.5.19.dev332 | shipped as local wheels and patches; the tags on the fork point at the base commit, not at the code |

## Verify a download

```bash
cd release-sm75-main && sha256sum -c SHA256SUMS && sha256sum -c MANIFEST.sha256
```

The wheel is byte-reproducible from the tag: `build-wheel.sh` in that directory
rebuilds it from a checkout of the tagged commit and produces the same sha256.
