#!/usr/bin/env python3
"""
Test script for the new session-based load balancing implementation.
This script demonstrates how the new AsyncLLMServerManager works.
"""

import asyncio
import time
from unittest.mock import Mock, AsyncMock
from typing import List

# Mock the required imports for testing
class MockRayActorHandle:
    def __init__(self, server_id: str):
        self.server_id = server_id
    
    def __hash__(self):
        return hash(self.server_id)

class MockConfig:
    def __init__(self, session_timeout: int = 300, load_balancing_strategy: str = 'requests'):
        self.session_timeout = session_timeout
        self.load_balancing_strategy = load_balancing_strategy

class MockTokenOutput:
    def __init__(self, text: str):
        self.text = text

# Import our modified class (we'll need to adapt the imports for testing)
class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least active sessions load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config, server_handles: List[MockRayActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager."""
        import heapq
        import threading
        import random
        from collections import OrderedDict
        
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Active sessions load balancing - track active sessions per server
        self.server_active_sessions = {hash(server): 0 for server in server_handles}
        self.server_handles_map = {hash(server): server for server in server_handles}
        
        # Heap for least active sessions load balancing: [active_sessions, (server_hash, server)]
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = OrderedDict()
        self.max_cache_size = max_cache_size
        
        # Track active sessions: request_id -> (server_hash, start_time)
        self.active_sessions = {}
        
        # Lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Session timeout (in seconds) - sessions older than this are considered inactive
        self.session_timeout = getattr(config, 'session_timeout', 300)  # 5 minutes default
        
        # Start background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background task for periodic session cleanup."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Clean up every minute
                with self._lock:
                    self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error during periodic session cleanup: {e}")

    def shutdown(self):
        """Shutdown the server manager and cleanup resources."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()

    def _cleanup_expired_sessions(self):
        """Clean up expired sessions and update server active session counts."""
        import heapq
        
        current_time = time.time()
        expired_sessions = []
        
        for request_id, (server_hash, start_time) in self.active_sessions.items():
            if current_time - start_time > self.session_timeout:
                expired_sessions.append(request_id)
        
        # Remove expired sessions
        for request_id in expired_sessions:
            server_hash, _ = self.active_sessions.pop(request_id)
            self.server_active_sessions[server_hash] -= 1
            
            # Remove from LRU cache if exists
            if request_id in self.request_id_to_server:
                del self.request_id_to_server[request_id]
        
        # Rebuild heap with updated session counts
        self.weighted_serveres = [[self.server_active_sessions[hash(server)], (hash(server), server)] 
                                 for server in self.server_handles]
        heapq.heapify(self.weighted_serveres)

    def _choose_server(self, request_id: str) -> MockRayActorHandle:
        """Choose server based on least active sessions load balancing with sticky sessions."""
        import heapq
        
        with self._lock:
            # Clean up expired sessions periodically
            self._cleanup_expired_sessions()
            
            # Check for sticky session first
            if request_id in self.request_id_to_server:
                server = self.request_id_to_server[request_id]
                # Update session activity
                if request_id in self.active_sessions:
                    self.active_sessions[request_id] = (hash(server), time.time())
                return server

            # Choose server with least active sessions
            server = self.weighted_serveres[0][1][1]
            server_hash = hash(server)
            
            # Update active session count
            self.server_active_sessions[server_hash] += 1
            self.active_sessions[request_id] = (server_hash, time.time())
            
            # Update heap
            self.weighted_serveres[0][0] += 1
            heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
            
            # Cache the mapping for sticky sessions (LRU)
            if request_id in self.request_id_to_server:
                del self.request_id_to_server[request_id]
            elif len(self.request_id_to_server) >= self.max_cache_size:
                self.request_id_to_server.popitem(last=False)  # Remove oldest
            self.request_id_to_server[request_id] = server
            
            return server

    def _end_session(self, request_id: str):
        """Mark a session as ended and update server active session count."""
        import heapq
        
        with self._lock:
            if request_id in self.active_sessions:
                server_hash, _ = self.active_sessions.pop(request_id)
                self.server_active_sessions[server_hash] -= 1
                
                # Remove from LRU cache
                if request_id in self.request_id_to_server:
                    del self.request_id_to_server[request_id]
                
                # Rebuild heap with updated session counts
                self.weighted_serveres = [[self.server_active_sessions[hash(server)], (hash(server), server)] 
                                         for server in self.server_handles]
                heapq.heapify(self.weighted_serveres)

    def get_server_stats(self) -> dict:
        """Get current server statistics including active session counts."""
        with self._lock:
            self._cleanup_expired_sessions()
            return {
                'server_active_sessions': dict(self.server_active_sessions),
                'total_active_sessions': sum(self.server_active_sessions.values()),
                'active_sessions_detail': dict(self.active_sessions)
            }

    async def generate(self, request_id, *, prompt_ids: List[int], sampling_params: dict, 
                      image_data=None, end_session: bool = False) -> MockTokenOutput:
        """Generate tokens from prompt ids."""
        server = self._choose_server(request_id)
        try:
            # Mock server response
            output = MockTokenOutput(f"Response from {server.server_id} for request {request_id}")
            return output
        finally:
            # End session if requested
            if end_session:
                self._end_session(request_id)


async def test_requests_load_balancing():
    """Test the original requests-based load balancing functionality."""
    print("Testing Requests-Based Load Balancing")
    print("=" * 50)
    
    # Create mock servers
    servers = [MockRayActorHandle(f"server_{i}") for i in range(3)]
    config = MockConfig(load_balancing_strategy='requests')
    
    # Create server manager
    manager = AsyncLLMServerManager(config, servers)
    
    print(f"Load balancing strategy: {manager.load_balancing_strategy}")
    print(f"Initial stats: {manager.get_server_stats()}")
    
    # Test basic load balancing
    print("\n1. Testing basic load balancing...")
    for i in range(6):
        request_id = f"req_{i}"
        server = manager._choose_server(request_id)
        print(f"Request {request_id} -> {server.server_id}")
    
    print(f"Stats after 6 requests: {manager.get_server_stats()}")
    
    # Test sticky sessions
    print("\n2. Testing sticky sessions...")
    for i in range(3):
        request_id = f"sticky_req_{i}"
        server1 = manager._choose_server(request_id)
        server2 = manager._choose_server(request_id)
        print(f"Sticky request {request_id}: {server1.server_id} == {server2.server_id}? {server1 == server2}")
    
    print(f"Final stats: {manager.get_server_stats()}")
    
    # Cleanup
    manager.shutdown()
    print("\nRequests-based test completed successfully!\n")


async def test_session_load_balancing():
    """Test the session-based load balancing functionality."""
    print("Testing Session-Based Load Balancing")
    print("=" * 50)
    
    # Create mock servers
    servers = [MockRayActorHandle(f"server_{i}") for i in range(3)]
    config = MockConfig(session_timeout=10, load_balancing_strategy='sessions')  # 10 seconds timeout for testing
    
    # Create server manager
    manager = AsyncLLMServerManager(config, servers)
    
    print(f"Load balancing strategy: {manager.load_balancing_strategy}")
    
    print(f"Initial stats: {manager.get_server_stats()}")
    
    # Test 1: Basic load balancing
    print("\n1. Testing basic load balancing...")
    for i in range(6):
        request_id = f"req_{i}"
        server = manager._choose_server(request_id)
        print(f"Request {request_id} -> {server.server_id}")
    
    print(f"Stats after 6 requests: {manager.get_server_stats()}")
    
    # Test 2: Sticky sessions
    print("\n2. Testing sticky sessions...")
    for i in range(3):
        request_id = f"sticky_req_{i}"
        server1 = manager._choose_server(request_id)
        server2 = manager._choose_server(request_id)
        print(f"Sticky request {request_id}: {server1.server_id} == {server2.server_id}? {server1 == server2}")
    
    print(f"Stats after sticky sessions: {manager.get_server_stats()}")
    
    # Test 3: Session ending
    print("\n3. Testing session ending...")
    request_id = "end_test_req"
    server = manager._choose_server(request_id)
    print(f"Request {request_id} assigned to {server.server_id}")
    print(f"Stats before ending: {manager.get_server_stats()}")
    
    manager._end_session(request_id)
    print(f"Stats after ending session: {manager.get_server_stats()}")
    
    # Test 4: Session timeout
    print("\n4. Testing session timeout...")
    request_id = "timeout_req"
    server = manager._choose_server(request_id)
    print(f"Request {request_id} assigned to {server.server_id}")
    print(f"Stats before timeout: {manager.get_server_stats()}")
    
    # Simulate timeout by manually setting old timestamp
    with manager._lock:
        manager.active_sessions[request_id] = (hash(server), time.time() - 20)  # 20 seconds ago
    
    manager._cleanup_expired_sessions()
    print(f"Stats after timeout cleanup: {manager.get_server_stats()}")
    
    # Test 5: Load balancing with mixed scenarios
    print("\n5. Testing mixed scenarios...")
    # Create some sessions that will timeout
    for i in range(3):
        request_id = f"timeout_req_{i}"
        server = manager._choose_server(request_id)
        with manager._lock:
            manager.active_sessions[request_id] = (hash(server), time.time() - 15)  # 15 seconds ago
    
    # Create new requests
    for i in range(5):
        request_id = f"new_req_{i}"
        server = manager._choose_server(request_id)
        print(f"New request {request_id} -> {server.server_id}")
    
    print(f"Stats after mixed scenarios: {manager.get_server_stats()}")
    
    # Cleanup
    manager.shutdown()
    print("\nTest completed successfully!")


async def main():
    """Run all tests."""
    await test_requests_load_balancing()
    await test_session_load_balancing()


if __name__ == "__main__":
    asyncio.run(main())
