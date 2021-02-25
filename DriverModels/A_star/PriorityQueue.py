import heapq


class PriorityQueue:
    """
    Priority queue for A-Star search. Contains all unexplored nodes with lowest priority node first.
    """
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority, scnd_priority):
        heapq.heappush(self.elements, (priority, scnd_priority, item))

    def get(self):
        return heapq.heappop(self.elements)[2]
