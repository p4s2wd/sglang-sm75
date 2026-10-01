# Production safety rules

This checkout is the **live production code**. The venv at
`/data/nvme/sglang/.venv` has sglang installed in editable mode against
`/data/nvme/sglang-codex/sglang/python`, so the working tree here *is* the code
the server runs. The launcher is `/data/nvme/sglang/deepseek-v4-flash.sh`.

## Never switch branches in this directory

`git checkout` here replaces the entire source tree under the running server.

Incident, 2026-10-02: `git checkout -b release/sm75main2` (branched from
`release/sm75main1`, a 2026-09-30 morning snapshot) removed
`python/sglang/kernels/ops/moe/moe_align.py`, which had arrived in the seven
commits made later that evening. That module is imported lazily, so startup
succeeded and the first request killed the server:

```
ModuleNotFoundError: No module named 'sglang.kernels.ops.moe.moe_align'
```

The follow-on `gloo ... Connection closed by peer` errors were cascade noise.
This was not an IMA regression.

## Rules

1. **Do not `git checkout` / `git switch` in this directory.** To look at or
   build another branch, use a separate worktree:

   ```sh
   git -C /data/nvme/sglang-codex/sglang worktree add /tmp/wt-sm75main1 <ref>
   ```

   or `git archive <ref> | tar -x -C /tmp/somewhere`. Never in place.

2. **Production branch is `sm75-dsv4-flash-main`.** `release/sm75mainN` branches
   are release snapshots: they carry the wheel and the release docs, and their
   source tree is *not* the development line.

3. **After anything that touches this working tree, verify the server** --
   a branch switch compiles fine and only shows up on the next request:

   ```sh
   pgrep -f '[b]in/sglang' | wc -l
   curl -s -m 5 localhost:8200/health
   ```

4. **The untracked `python/`, `rust/`, `test/` trees are build residue** from
   `build-wheel.sh` (2.1 GB of rust `target/`). They are not source. Leave them
   alone and never `git add -A` here.
