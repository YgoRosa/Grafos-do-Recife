import heapq
from collections import deque
from typing import List, Dict, Tuple, Union, Any, Optional
from .graph import Graph

class PositiveFloat(float):
    def __new__(cls, value):
        if value < 0:
            raise ValueError("Dijkstra does not accept negative weights.")
        return super(PositiveFloat, cls).__new__(cls, value)

class Algorithms:
    @staticmethod
    def dijkstra(graph: Graph, start: str, end: str) -> Dict[str, Any]:
        if not graph.has_node(start) or not graph.has_node(end):
            return {"cost": float('inf'), "path": [], "error": "Invalid nodes"}

        distances = {node: float('inf') for node in graph.get_nodes()}
        distances[start] = 0.0
        predecessors = {start: None}
        visited = set()
        
        pq: List[Tuple[float, str]] = []
        heapq.heappush(pq, (0.0, start))
        
        visited_count = 0

        while pq:
            d_curr, u = heapq.heappop(pq)
            
            if u == end:
                break

            if d_curr > distances[u]:
                continue
            
            visited.add(u)
            visited_count += 1
            
            for v in graph.neighbors(u):
                if v in visited: 
                    continue

                edge_data = graph.get_edge_data(u, v)
                if not edge_data: continue
                
                raw_weight = edge_data[0]
                
                try:
                    weight = PositiveFloat(raw_weight)
                except ValueError:
                    continue 

                new_dist = distances[u] + weight
                
                if new_dist < distances[v]:
                    distances[v] = new_dist
                    predecessors[v] = u
                    heapq.heappush(pq, (new_dist, v))

        path = []
        curr = end
        
        if distances[end] == float('inf'):
             return {"cost": float('inf'), "path": [], "visited_count": visited_count}

        while curr is not None:
            path.append(curr)
            curr = predecessors.get(curr)
            if len(path) > len(graph): break 
            
        path.reverse()

        if not path or path[0] != start:
            return {"cost": float('inf'), "path": [], "visited_count": visited_count}

        return {
            "cost": distances[end],
            "path": path,
            "visited_count": visited_count
        }

    @staticmethod
    def bellman_ford(graph: Graph, start: str, end: str) -> Dict[str, Any]:
        if not graph.has_node(start):
            return {"cost": float('inf'), "path": [], "negative_cycle": False}

        nodes = graph.get_nodes()
        distances = {node: float('inf') for node in nodes}
        distances[start] = 0.0
        predecessors = {start: None}

        all_edges = []
        for u, v, w, _ in graph.get_edges():
            all_edges.append((u, v, w))

        for _ in range(len(nodes) - 1):
            relaxed = False
            for u, v, w in all_edges:
                if distances[u] != float('inf') and distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
                    predecessors[v] = u
                    relaxed = True

            if not relaxed:
                break

        has_negative_cycle = False
        for u, v, w in all_edges:
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                has_negative_cycle = True
                break

        path = []
        if not has_negative_cycle and end in distances and distances[end] != float('inf'):
            curr = end
            path_set = set()
            
            while curr is not None:
                if curr in path_set: break 
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
        if not graph.has_node(start):
            return {"error": f"Node {start} does not exist"}

        state = {u: 'white' for u in graph.get_nodes()}
        discovery_time = {}
        finish_time = {}
        predecessors = {u: None for u in graph.get_nodes()}
        
        edge_classification = {}
        visit_order = []
        time_counter = 0
        has_cycle = False
        
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

        dfs_visit(start)
        
        return {
            "order": visit_order,
            "has_cycle": has_cycle,
            "classification": edge_classification,
            "times": {"discovery": discovery_time, "finish": finish_time}
        }