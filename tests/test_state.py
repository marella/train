from collections import deque

import pytest
from train import State, RingBuffer, Transition, Transitions


class TestRingBuffer():

    @pytest.mark.parametrize('maxlen', [-1, 1, 5])
    def test_ops(self, maxlen):
        q = deque(maxlen=None if maxlen < 0 else maxlen)
        b = RingBuffer(maxlen=maxlen)
        self.assert_empty(b)
        limit = 5 if maxlen < 0 else 2 * maxlen
        for i in range(limit):
            q.append(i)
            b.append(i)
            data = list(q)
            l = len(b)
            s = b.sample(l)
            assert l == len(q)
            assert b.get() == data
            assert b.last() == q[-1]
            assert len(s) == l
            assert sorted(s) == data
            with pytest.raises(ValueError):
                b.sample(l + 1)
        b.reset()
        self.assert_empty(b)

    def assert_empty(self, b):
        assert len(b) == 0
        assert b.get() == []
        with pytest.raises(IndexError):
            b.last()
