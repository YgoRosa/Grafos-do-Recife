import heapq
from collections import deque
from typing import List, Dict, Tuple, Union, Any, Optional

# Adjust import according to your folder structure
from .graph import Graph

class PositiveFloat(float):
    """
    Wrapper to enforce non-negative weights for Dijkstra's algorithm.
    """
    def __new__(cls, value):
        if value < 0:
            raise ValueError("Dijkstra does not accept negative weights.")
        return super(PositiveFloat, cls).__new__(cls, value)

class Algorithms:
    """
    Static implementation of classic Graph algorithms.
    Contains: Dijkstra, Bellman-Ford, BFS, DFS.
    """

    @staticmethod
    def dijkstra(graph: Graph, start: str, end: str) -> Dict[str, Any]:
        """
        Calculates the Shortest Path using Dijkstra's Algorithm (Binary Heap).
        
        Args:
            graph: Graph instance.
            start: Source node ID.
            end: Target node ID.
            
        Returns:
            Dict containing: 'cost', 'path' (list of nodes), 'visited_count'.
        """
        if not graph.has_node(start) or not graph.has_node(end):
            return {"cost": float('inf'), "path": [], "error": "Invalid nodes"}

        # Initialization
        distances = {node: float('inf') for node in graph.get_nodes()}
        distances[start] = 0.0
        predecessors = {start: None}
        visited = set()
        
        # Priority Queue: (cost, current_node)
        pq: List[Tuple[float, str]] = []
        heapq.heappush(pq, (0.0, start))
        
        visited_count = 0

        while pq:
            d_curr, u = heapq.heappop(pq)
            
            # Optimization: Stop if target is reached
            if u == end:
                break

            if d_curr > distances[u]:
                continue
            
            visited.add(u)
            visited_count += 1
            
            for v in graph.neighbors(u):
                if v in visited: 
                    continue

                # Safely get edge data
                edge_data = graph.get_edge_data(u, v)
                if not edge_data: continue
                
                raw_weight = edge_data[0]
                
                try:
                    # Enforce positive weight constraint
                    weight = PositiveFloat(raw_weight)
                except ValueError:
                    continue # Ignore negative edges in Dijkstra

                new_dist = distances[u] + weight
                
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    predecessors[v] = u
                    heapq.heappush(pq, (new_dist, v))

        # Path Reconstruction
        path = []
        curr = end
        
        # If target unreachable
        if distances[end] == float('inf'):
             return {"cost": float('inf'), "path": [], "visited_count": visited_count}

        # Backtrack from end to start
        while curr is not None:
            path.append(curr)
            curr = predecessors.get(curr)
            # Safety break for loops
            if len(path) > len(graph): break 
            
        path.reverse()

        # Validate path origin
        if not path or path[0] != start:
            return {"cost": float('inf'), "path": [], "visited_count": visited_count}

        return {
            "cost": distances[end],
            "path": path,
            "visited_count": visited_count
        }

    @staticmethod
    def bellman_ford(graph: Graph, start: str, end: str) -> Dict[str, Any]:
        """
        Calculates Shortest Path using Bellman-Ford.
        Supports negative weights and detects negative cycles.
        
        Returns:
            Dict containing: 'cost', 'path', 'negative_cycle' (bool).
        """
        if not graph.has_node(start):
            return {"cost": float('inf'), "path": [], "negative_cycle": False}

        nodes = graph.get_nodes()
        distances = {node: float('inf') for node in nodes}
        distances[start] = 0.0
        predecessors = {start: None}
        
        # Flatten all edges for iteration: List of (u, v, weight)
        all_edges = []
        for u, v, w, _ in graph.get_edges():
            all_edges.append((u, v, w))

        # 1. Relaxation Phase (V-1 times)
        for _ in range(len(nodes) - 1):
            relaxed = False
            for u, v, w in all_edges:
                if distances[u] != float('inf') and distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
                    predecessors[v] = u
                    relaxed = True
            
            # Optimization: Early stopping if no changes occurred (crucial for positive graphs)
            if not relaxed:
                break

        # 2. Negative Cycle Detection Phase
        has_negative_cycle = False
        for u, v, w in all_edges:
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                has_negative_cycle = True
                break
        
        # Path Reconstruction
        path = []
        # Only reconstruct if no negative cycle and target is reachable
        if not has_negative_cycle and end in distances and distances[end] != float('inf'):
            curr = end
            path_set = set()
            
            while curr is not None:
                if curr in path_set: break # Avoid infinite loops
                path_set.add(curr)
                path.append(curr)
                curr = predecessors.get(curr)
                
            path.reverse()
            
            if not path or path[0] != start:
                path = []

        return {
            "cost": distances.get(end, float('inf')),
            "path": path,
            "negative_cycle": has_negative_cycle
        }

    @staticmethod
    def bfs(graph: Graph, start: str) -> Dict[str, Any]:
        """
        Breadth-First Search.
        Returns topological levels (distance in hops) and visitation order.
        """
        if not graph.has_node(start):
            return {"error": f"Node {start} does not exist"}

        visited = {start}
        queue = deque([start])
        
        predecessors = {start: None}
        levels = {start: 0} 
        visit_order = []
        
        while queue:
            u = queue.popleft()
            visit_order.append(u)
            
            for v in graph.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    predecessors[v] = u
                    levels[v] = levels[u] + 1
                    queue.append(v)
        
        return {
            "levels": levels,
            "order": visit_order,
            "parents": predecessors
        }

    @staticmethod
    def dfs(graph: Graph, start: str) -> Dict[str, Any]:
        """
        Depth-First Search (Iterative).
        Performs edge classification (Tree, Back, Forward, Cross) and detects cycles.
        """
        if not graph.has_node(start):
            return {"error": f"Node {start} does not exist"}

        # State tracking: 'white' (unvisited), 'gray' (visiting), 'black' (finished)
        state = {u: 'white' for u in graph.get_nodes()}
        discovery_time = {}
        finish_time = {}
        predecessors = {u: None for u in graph.get_nodes()}
        
        edge_classification = {} # (u, v) -> type
        visit_order = []
        time_counter = 0
        has_cycle = False
        
        # Iterative DFS using a manual stack to simulate recursion
        # Stack elements: (node, iterator_of_neighbors)
        # We need to peek at stack to mark finish time (post-order)
        
        # Simplified recursive wrapper for clarity in edge classification logic
        # (Python recursion limit is usually 1000, watch out for very deep graphs)
        # Switching to recursive for correct Edge Classification logic as requested in requirements
        
        def dfs_visit(u: str):
            nonlocal time_counter, has_cycle
            
            state[u] = 'gray'
            time_counter += 1
            discovery_time[u] = time_counter
            visit_order.append(u)
            
            for v in graph.neighbors(u):
                edge = (u, v)
                
                if state[v] == 'white':
                    edge_classification[edge] = 'tree'
                    predecessors[v] = u
                    dfs_visit(v)
                
                elif state[v] == 'gray':
                    edge_classification[edge] = 'back'
                    has_cycle = True 
                
                elif state[v] == 'black':
                    if discovery_time[u] < discovery_time[v]:
                        edge_classification[edge] = 'forward'
                    else:
                        edge_classification[edge] = 'cross'

            state[u] = 'black'
            time_counter += 1
            finish_time[u] = time_counter

        # Start DFS
        dfs_visit(start)
        
        return {
            "order": visit_order,
            "has_cycle": has_cycle,
            "classification": edge_classification,
            "times": {"discovery": discovery_time, "finish": finish_time}
        }