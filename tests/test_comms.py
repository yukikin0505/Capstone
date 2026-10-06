import numpy as np

from swarm import comms


def test_unit_disk_links_and_weights():
    pos = np.array([[0.0, 0.0], [30.0, 0.0], [100.0, 0.0]])
    g = comms.make_graph(pos, r_c=40.0, sigma=40.0)
    A = g.A
    assert A[0, 1] == A[1, 0] == np.exp(-(30.0**2) / 40.0**2)  # inside r_c
    assert A[0, 2] == A[1, 2] == 0.0  # outside r_c: no link
    assert np.all(np.diag(A) == 0)  # no self-loops
    assert len(g.edges) == 2  # 1 link = 2 directed edges
    assert not comms.is_connected(g)


def test_random_graph_is_connected_and_reproducible():
    def draw():
        return comms.random_connected_graph(8, 100.0, 40.0, 40.0, np.random.default_rng(1), 10_000)

    g1, g2 = draw(), draw()
    assert comms.is_connected(g1)
    assert np.array_equal(g1.pos, g2.pos)


def test_deliver_only_neighbours():
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [90.0, 0.0]])
    g = comms.make_graph(pos, r_c=40.0, sigma=40.0)
    broadcast = np.array([1.0, 2.0, 3.0])
    assert [x for _, x in comms.deliver(g, 0, broadcast)] == [2.0]  # not from drone 2
    assert comms.deliver(g, 2, broadcast) == []
