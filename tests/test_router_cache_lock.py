import threading
import time
import sys
sys.stdout.reconfigure(encoding='utf-8')
from models.security.v61_inference_router import V61SecurityRouter

def test_singleton_thread_safety():
    routers = []
    def init_router():
        # Gọi __init__ của Singleton
        router = V61SecurityRouter(ollama_model="gemma3:4b")
        routers.append(router)

    threads = []
    print("Bắt đầu khởi tạo đồng thời từ 10 threads...")
    for _ in range(10):
        t = threading.Thread(target=init_router)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Kiểm tra xem tất cả có phải là cùng 1 instance không
    first = routers[0]
    success = all(r is first for r in routers)
    print(f"Kiểm tra Singleton instance identity: {'PASS' if success else 'FAIL'}")
    
    # Kiểm tra LRU Cache
    print("\nKiểm tra LRU cache...")
    t0 = time.perf_counter()
    res1 = first.model_a.score("liệt kê thư mục")
    t1 = time.perf_counter()
    res2 = first.model_a.score("liệt kê thư mục")
    t2 = time.perf_counter()
    
    # Cache hit thì phải nhanh hơn nhiều
    first_call = (t1 - t0) * 1000
    second_call = (t2 - t1) * 1000
    print(f"Gọi lần 1 (Miss): {first_call:.2f} ms")
    print(f"Gọi lần 2 (Hit) : {second_call:.2f} ms")
    
    if second_call < first_call * 0.1:
        print("LRU Cache: PASS")
    else:
        print("LRU Cache: FAIL (không nhanh hơn đáng kể)")
        
    if success and (second_call < first_call * 0.1):
        print("\nTẤT CẢ TEST PASS!")
        sys.exit(0)
    else:
        print("\nCÓ LỖI XẢY RA!")
        sys.exit(1)

if __name__ == "__main__":
    test_singleton_thread_safety()
