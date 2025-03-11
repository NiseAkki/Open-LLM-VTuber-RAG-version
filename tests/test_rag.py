import pytest
import sys
import os
from pathlib import Path
from open_llm_vtuber.rag import RAGSystem
from open_llm_vtuber.rag.config import load_config

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

@pytest.fixture
def rag_system():
    """初始化RAG系统"""
    # 获取测试配置文件的绝对路径
    config_path = os.path.join(project_root, "tests", "test_config.yaml")
    system = RAGSystem(config_path)
    system.initialize()
    return system

def test_rag_initialization(rag_system):
    """测试RAG系统初始化"""
    assert rag_system.retriever is not None
    assert rag_system.loader is not None
    assert len(rag_system.loader.vault_content) >= 2

def test_corpus_loading(rag_system):
    """验证语料库加载完整性"""
    expected_contents = [
        "赞赞的宠物叫哔哔",
        "赞赞的爸爸叫余昊"
    ]
    
    # 检查是否包含所有预期内容
    for content in expected_contents:
        assert any(content in line for line in rag_system.loader.vault_content)

def test_query_retrieval(rag_system):
    """测试基本查询检索"""
    test_cases = [
        {
            "input": "赞赞的宠物叫什么？",
            "expected_keywords": ["哔哔"]
        },
        {
            "input": "余昊是谁的爸爸？",
            "expected_keywords": ["赞赞"]
        }
    ]
    
    for case in test_cases:
        response = rag_system.query(case["input"])
        print(f"Query: {case['input']}")
        print(f"Response: {response}")

        # 验证响应包含关键词
        assert any(keyword in response for keyword in case["expected_keywords"])

def test_context_aware_query(rag_system):
    """测试上下文感知查询"""
    conversation_history = [
        {"role": "user", "content": "谁有宠物？"},
        {"role": "assistant", "content": "赞赞有宠物"}
    ]
    
    response = rag_system.query(
        "它叫什么名字？",
        conversation_history=conversation_history
    )
    
    assert "哔哔" in response
    assert len(response) >= 15

def test_edge_cases(rag_system):
    """测试边界情况"""
    # 测试无关查询
    response = rag_system.query("今天的天气如何？")
    assert "不知道" in response or "不相关" in response
    
    # 测试空输入
    with pytest.raises(ValueError, match="查询内容不能为空"):
        rag_system.query("")
    
    # 测试空白字符输入
    with pytest.raises(ValueError, match="查询内容不能为空"):
        rag_system.query("   ")

def test_embedding_cache(rag_system):
    """测试嵌入向量缓存机制"""
    # 连续两次查询相同的内容
    query = "赞赞的宠物叫什么？"
    
    # 第一次查询
    response1 = rag_system.query(query)
    assert response1 is not None
    
    # 第二次查询应该使用缓存
    response2 = rag_system.query(query)
    assert response2 is not None
    
    # 两次响应应该一致
    assert response1 == response2

def test_retry_mechanism(rag_system):
    """测试重试机制"""
    # 测试健康检查
    assert rag_system.retriever.check_health()
    
    # 测试错误重试（通过健康检查函数）
    try:
        response = rag_system.query("测试查询")
        assert response is not None
        assert len(response) > 0
    except Exception as e:
        pytest.fail(f"重试机制失败: {str(e)}")

def test_health_check(rag_system):
    """测试健康检查功能"""
    # 验证初始健康状态
    assert rag_system.retriever.check_health()
    
    # 验证语料库状态
    assert len(rag_system.loader.vault_content) > 0
    assert rag_system.loader.vault_embeddings is not None
    assert rag_system.loader.vault_embeddings.shape[0] == len(rag_system.loader.vault_content)

if __name__ == "__main__":
    pytest.main(["-v", __file__])