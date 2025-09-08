# 负载均衡实现说明

## 概述

在 `AsyncLLMServerManager` 中实现了两种负载均衡策略，通过配置开关可以选择使用哪种策略。

## 负载均衡策略

### 1. Requests-based Load Balancing (默认)
- **策略名称**: `requests`
- **工作原理**: 基于每个server的总请求数进行负载均衡
- **选择逻辑**: 总是选择总请求数最少的server
- **适用场景**: 无状态工作负载，请求处理时间相对均匀
- **优势**: 实现简单，开销低，适合大多数场景

### 2. Sessions-based Load Balancing (新增)
- **策略名称**: `sessions`
- **工作原理**: 基于每个server的活跃会话数进行负载均衡
- **选择逻辑**: 总是选择活跃会话数最少的server
- **适用场景**: 有状态工作负载，长时间运行的会话
- **优势**: 更精确的负载控制，支持会话超时和清理

## 配置方式

### 在配置文件中设置

```yaml
# 使用requests-based负载均衡 (默认)
load_balancing_strategy: "requests"

# 或使用sessions-based负载均衡
load_balancing_strategy: "sessions"
session_timeout: 300  # 会话超时时间(秒)，默认300秒
```

### 在代码中设置

```python
config = DictConfig({
    'load_balancing_strategy': 'sessions',  # 或 'requests'
    'session_timeout': 600  # 可选，仅对sessions策略有效
})

manager = AsyncLLMServerManager(config, server_handles)
```

## 功能特性

### 共同特性
- **粘性会话**: 相同`request_id`的请求会路由到同一个server
- **LRU缓存**: 使用LRU缓存管理request_id到server的映射
- **线程安全**: 所有操作都是线程安全的

### Sessions策略特有特性
- **会话跟踪**: 跟踪每个活跃会话的开始时间
- **自动清理**: 定期清理超时的会话
- **会话结束**: 支持手动结束会话
- **后台任务**: 每分钟自动清理过期会话

## API变化

### generate方法
```python
async def generate(
    self,
    request_id,
    *,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    image_data: Optional[list[Any]] = None,
    end_session: bool = False,  # 新增参数
) -> TokenOutput:
```

- `end_session`参数仅在sessions策略下有效
- 当`end_session=True`时，请求完成后会结束该会话

### 统计信息
```python
stats = manager.get_server_stats()
```

**Requests策略返回**:
```python
{
    'load_balancing_strategy': 'requests',
    'server_total_requests': {server_hash: count, ...},
    'total_requests': total_count
}
```

**Sessions策略返回**:
```python
{
    'load_balancing_strategy': 'sessions',
    'server_active_sessions': {server_hash: count, ...},
    'total_active_sessions': total_count,
    'active_sessions_detail': {request_id: (server_hash, start_time), ...}
}
```

## 性能考虑

### Requests策略
- **内存开销**: 低，只跟踪请求计数
- **CPU开销**: 低，简单的堆操作
- **网络开销**: 无额外开销

### Sessions策略
- **内存开销**: 中等，需要跟踪活跃会话详情
- **CPU开销**: 中等，包括定期清理任务
- **网络开销**: 无额外开销

## 使用建议

### 选择Requests策略的场景
- 无状态应用
- 请求处理时间相对均匀
- 对性能要求极高
- 不需要会话管理

### 选择Sessions策略的场景
- 有状态应用
- 长时间运行的会话
- 需要精确的负载控制
- 需要会话生命周期管理

## 向后兼容性

- 默认使用requests策略，保持原有行为
- 现有代码无需修改即可继续工作
- 新功能通过配置开关启用

## 测试

运行测试脚本验证两种策略:
```bash
python test_session_load_balancing.py
```

测试包括:
- 基本负载均衡
- 粘性会话
- 会话结束
- 会话超时
- 混合场景
