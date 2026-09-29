import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _StopAfterOneIteration(Exception):
    pass


class _Event:
    def synchronize(self):
        return None


class _Stream:
    def wait_event(self, event):
        return None


class _Device:
    @staticmethod
    def current_stream():
        return _Stream()


class _Plan:
    running_batch = object()
    batch_to_run = object()


class _Scheduler(SchedulerPPMixin):
    def __init__(self, calls):
        self.calls = calls
        self.pp_loop_size = 1
        self.ps = SimpleNamespace(pp_size=1)
        self.pp_group = SimpleNamespace(is_last_rank=False)
        self.request_receiver = SimpleNamespace(recv_requests=lambda: [])
        self.mbs = [object()]
        self.running_mbs = [None]
        self.last_mbs = [None]
        self.send_req_work = None
        self.send_proxy_work = None
        self.send_proxy_requires_forward_fence = False
        self.mb_metadata = [None]
        self.last_rank_comm_queue = None
        self.device_module = _Device
        self.iterations = 0

    def init_pp_loop_state(self):
        return None

    def ingest_requests(self):
        return []

    def process_input_requests(self, requests):
        return None

    def _pp_commit_comm_work(self, work, fence_next_forward=False):
        return None

    def _pp_send_pyobj_to_next_stage(self, requests, async_send):
        return None

    def get_next_batch_to_run(self, running_batch, last_batch):
        if self.iterations:
            raise _StopAfterOneIteration
        self.iterations += 1
        return _Plan()

    def _pp_recv_proxy_tensors(self):
        return object()

    def _pp_commit_send_output_work_and_preprocess_output_tensors(
        self, next_first_rank_mb_id, next_mb_id
    ):
        return None, None, _Event()

    def _pp_launch_batch(self, mb_id, batch, proxy, metadata, queue):
        result = SimpleNamespace(
            pp_hidden_states_proxy_tensors=SimpleNamespace(tensors=object()),
            can_run_cuda_graph=False,
        )
        return result, object()

    def _pp_process_batch_result(self, batch, result):
        self.calls.append("process")

    def _pp_send_dict_to_next_stage(
        self, tensors, async_send, msg_type, ready_event=None
    ):
        self.calls.append("send")
        return object()

    def on_idle(self):
        raise _StopAfterOneIteration


class TestSchedulerPPProxyOrder(unittest.TestCase):
    def _run(self, early):
        calls = []
        with (
            envs.SGLANG_PP_EARLY_PROXY_SEND.override(early),
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                return_value=SimpleNamespace(pp_size=1, pp_async_batch_depth=1),
            ),
            self.assertRaises(_StopAfterOneIteration),
        ):
            _Scheduler(calls).event_loop_pp()
        return calls

    def test_early_proxy_send_precedes_result_processing(self):
        self.assertEqual(self._run(True), ["send", "process"])

    def test_default_order_processes_result_before_proxy_send(self):
        self.assertEqual(self._run(False), ["process", "send"])


if __name__ == "__main__":
    unittest.main()
