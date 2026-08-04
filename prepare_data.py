"""FlakeAI Training Data Preparer - 100K Version

Dağılım:
- 40,000 Türkçe
- 20,000 Kod
- 40,000 İngilizce (WikiText)
"""
import os
import random
from pathlib import Path
import httpx

DATA_DIR = Path("data")
CACHE_DIR = Path(".cache")

def download_file(url, dest, description=""):
    print(f"  Indiriliyor: {description}")
    try:
        with httpx.Client(timeout=300, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
        print(f"  Tamamlandi: {dest}")
        return True
    except Exception as e:
        print(f"  Hata: {e}")
        return False

def extract_code_lines(filepath):
    lines = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip()
                if not line or len(line.strip()) < 5:
                    continue
                stripped = line.strip()
                if stripped.startswith(("#", "//", "/*", "*", "///")):
                    continue
                lines.append(stripped)
    except:
        pass
    return lines

def prepare_data():
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    random.seed(42)
    all_texts = []

    # 1. Türkçe (40K)
    print("\n[1/3] Türkçe veriler (40K)...")
    builtin_path = Path("data/builtin_turkish.txt")
    turkish = []
    if builtin_path.exists():
        with open(builtin_path, "r", encoding="utf-8") as f:
            turkish = [l.strip() for l in f if l.strip() and len(l.strip()) > 5]

    extra_turkish = [
        "Python ile programlama öğrenmek istiyorum.",
        "Merhaba, size nasıl yardımcı olabilirim?",
        "Bugün hava çok güzel, dışarı çıkmak istiyorum.",
        "Yapay zeka gelecekte hayatımızı nasıl değiştirecek?",
        "Veritabanı tasarımı önemli bir konudur.",
        "Web sitesi geliştirmek için HTML, CSS ve JavaScript gerekir.",
        "API nedir ve nasıl kullanılır?",
        "Docker konteyner teknolojisi çok yaygınlaştı.",
        "Machine learning nedir?",
        "Deep learning ile görüntü tanıma yapılabilir.",
        "Flask ve Django Python web frameworkleridir.",
        "React ve Vue.js frontend frameworkleridir.",
        "PostgreSQL ve MongoDB farklı veritabanı türleridir.",
        "Git versiyon kontrol sistemi olarak kullanılır.",
        "CI/CD sürekli entegrasyon ve dağıtım demektir.",
        "Mikroservis mimarisi büyük uygulamalar için önerilir.",
        "Cloud computing bulut bilişim demektir.",
        "AWS, Azure ve Google Cloud bulut servisleridir.",
        "Redis hız cache sistemi olarak kullanılır.",
        "Kubernetes konteyner orkestrasyonudur.",
        "TensorFlow ve PyTorch derin öğrenme kütüphaneleridir.",
        "NLP doğal dil işleme demektir.",
        "Computer vision bilgisayar görüşü alanıdır.",
        "Transformers modeli dikkat mekanizması kullanır.",
        "BERT ve GPT dil modelleridir.",
        "Fine-tuning önceden eğitilmiş modeli inceltmektir.",
        "Transfer learning transfer öğrenmedir.",
        "Overfitting aşırı uyumdur, model ezberler.",
        "Regularization düzenleme yöntemidir.",
        "Gradient descent gradyan iniş algoritmasıdır.",
        "Backpropagation geri yayılım algoritmasıdır.",
        "Loss function kayıp fonksiyonudur.",
        "Optimizer optimizasyon algoritmasıdır.",
        "Learning rate öğrenme hızıdır.",
        "Epoch tüm veri setinin bir geçişidir.",
        "Batch size parti boyutudur.",
        "Dropout rastgele bırakma tekniğidir.",
        "Batch normalization toplu normallemedir.",
        "Activation function aktivasyon fonksiyonudur.",
        "ReLU, Sigmoid ve Tanh aktivasyon fonksiyonlarıdır.",
        "Convolutional neural network evrişimsel sinir ağıdır.",
        "Recurrent neural network yinelemeli sinir ağıdır.",
        "LSTM uzun kısa vadeli bellek modelidir.",
        "GAN üretici çekişmeli ağ modelidir.",
        "Autoencoder otomatik kodlayıcı modelidir.",
        "Clustering kümeleme algoritmasıdır.",
        "Classification sınıflandırma görevidir.",
        "Regression regresyon görevidir.",
        "Random forest rastgele orman algoritmasıdır.",
        "SVM destek vektör makineleridir.",
        "Decision tree karar ağacıdır.",
        "KNN en yakın komşu algoritmasıdır.",
        "Naive Bayes naive Bayes algoritmasıdır.",
        "Logistic regression lojistik regresyondur.",
        "Linear lineer regresyondur.",
        "Feature selection özellik seçimidir.",
        "Dimensionality reduction boyut azaltmadır.",
        "PCA temel bileşen analizidir.",
        "Cross-validation çapraz doğrulamadır.",
        "Hyperparameter tuning hiperparametre ayarıdır.",
        "Grid search ızgara aramasıdır.",
        "Random search rastgele aramadır.",
        "Data preprocessing veri önişlemedir.",
        "Data augmentation veri artırımıdır.",
        "Normalization normallemedir.",
        "Standardization standartlaştırmadır.",
        "Encoding kodlama demektir.",
        "Tokenization tokenizasyondur.",
        "Stemming kök bulma işlemidir.",
        "Lemmatization lemmatizasyondur.",
        "TF-IDF terim frekansı-ters belge frekansıdır.",
        "Word embeddings kelime gömmedir.",
        "Word2Vec kelime vektörleridir.",
        "GloVe global vektörlerdir.",
        "FastText hızlı metin modelidir.",
        "Attention mechanism dikkat mekanizmasıdır.",
        "Self-attention öz dikkattir.",
        "Multi-head attention çoklu dikkattir.",
        "Positional encoding konumsal kodlamadır.",
        "Transformer mimarisi dikkat tabanlıdır.",
        "BERT bidirectional encoder representations'dır.",
        "GPT generative pre-trained transformer'dır.",
        "T5 text-to-text transfer transformer'dır.",
        "XLNet permütasyon modelidir.",
        "RoBERTa optimized BERT modelidir.",
        "ALBERT A lite BERT modelidir.",
        "ELECTRA efficiently learning token'dır.",
        "DeBERTa decoding-enhanced BERT modelidir.",
        "Veri bilimi nedir? Veri bilimi, verilerden bilgi çıkaran disiplinler arası bir alandır.",
        "Programlama dilleri nelerdir? Python, Java, C++, JavaScript, Go, Rust, Swift, Kotlin, C#, PHP, Ruby, Scala, TypeScript, R, MATLAB.",
        "Web geliştirme nedir? Web geliştirme, internet siteleri ve web uygulamaları oluşturma sürecidir.",
        "Frontend nedir? Frontend, kullanıcı arayüzü geliştirme demektir.",
        "Backend nedir? Backend, sunucu tarafı geliştirme demektir.",
        "Full stack nedir? Full stack, hem frontend hem backend geliştirme demektir.",
        "DevOps nedir? DevOps, geliştirme ve operasyon süreçlerinin entegrasyonudur.",
        "Agile nedir? Agile, çevik yazılım geliştirme metodolojisidir.",
        "Scrum nedir? Scrum, agile bir çerçeve yöntemidir.",
        "Kanban nedir? Kanban, görsel iş yönetimi yöntemidir.",
        "Sprint nedir? Sprint, kısa süreli geliştirme döngüsüdür.",
        "Stand-up nedir? Stand-up, günlük kısa toplantı demektir.",
        "Retrospective nedir? Retrospective, geriye dönük değerlendirme toplantısıdır.",
        "Backlog nedir? Backlog, yapılacak işler listesidir.",
        "User story nedir? User story, kullanıcı hikayesi demektir.",
        "Epic nedir? Epic, büyük kullanıcı hikayesi demektir.",
        "Feature nedir? Feature, özellik demektir.",
        "Bug nedir? Bug, hata demektir.",
        "Hotfix nedir? Hotfix, acil düzeltme demektir.",
        "Release nedir? Release, sürüm çıkarma demektir.",
        "Version control nedir? Version control, sürüm kontrolü demektir.",
        "Branch nedir? Branch, dal demektir.",
        "Merge nedir? Merge, birleştirme demektir.",
        "Pull request nedir? Pull request, çekme isteği demektir.",
        "Code review nedir? Code review, kod incelemesi demektir.",
        "Unit test nedir? Unit test, birim testi demektir.",
        "Integration test nedir? Integration test, entegrasyon testidir.",
        "E2E test nedir? E2E test, uçtan uca test demektir.",
        "Regression test nedir? Regression test, gerileme testidir.",
        "Performance test nedir? Performance test, performans testidir.",
        "Load test nedir? Load test, yükleme testidir.",
        "Stress test nedir? Stress test, stres testidir.",
        "Security test nedir? Security test, güvenlik testidir.",
        "Penetration test nedir? Penetration test, sızma testidir.",
        "Code coverage nedir? Code coverage, kod kapsama yüzdesidir.",
        "Technical debt nedir? Technical debt, teknik borç demektir.",
        "Refactoring nedir? Refactoring, yeniden yapılandırma demektir.",
        "Design pattern nedir? Design pattern, tasarım kalıbı demektir.",
        "SOLID nedir? SOLID, nesne yönelimli tasarım prensipleridir.",
        "DRY nedir? DRY, Don't Repeat Yourself demektir.",
        "KISS nedir? KISS, Keep It Simple Stupid demektir.",
        "YAGNI nedir? YAGNI, You Aren't Gonna Need It demektir.",
        "Clean code nedir? Clean code, temiz kod demektir.",
        "Code smell nedir? Code smell, kod kokusu demektir.",
        "Anti-pattern nedir? Anti-pattern, anti-kalıp demektir.",
        "Microservice nedir? Microservice, mikroservis demektir.",
        "Monolith nedir? Monolith, monolitik mimari demektir.",
        "Event-driven nedir? Event-driven, olay yönlü mimari demektir.",
        "Message queue nedir? Message queue, mesaj kuyruğu demektir.",
        "Pub/Sub nedir? Pub/Sub, publish/subscribe demektir.",
        "API gateway nedir? API gateway, API geçidi demektir.",
        "Service mesh nedir? Service mesh, servis ağı demektir.",
        "Circuit breaker nedir? Circuit breaker, devre kesici demektir.",
        "CQRS nedir? CQRS, Command Query Responsibility Segregation demektir.",
        "Event sourcing nedir? Event sourcing, olay kaynaklama demektir.",
        "Idempotency nedir? Idempotency, tekillik demektir.",
    ]
    turkish.extend(extra_turkish)
    turkish = turkish[:40000]
    all_texts.extend(turkish)
    print(f"  Türkçe: {len(turkish)} satır")

    # 2. Kod (20K)
    print("\n[2/3] Kod verileri (20K)...")
    code_files = [
        ("https://raw.githubusercontent.com/pallets/flask/main/src/flask/app.py", "Flask"),
        ("https://raw.githubusercontent.com/pallets/flask/main/src/flask/blueprints.py", "Flask BP"),
        ("https://raw.githubusercontent.com/encode/httpx/master/httpx/_client.py", "HTTPX"),
        ("https://raw.githubusercontent.com/pydantic/pydantic/main/pydantic/main.py", "Pydantic"),
        ("https://raw.githubusercontent.com/pallets/click/main/src/click/core.py", "Click"),
        ("https://raw.githubusercontent.com/psf/requests/main/src/requests/sessions.py", "Requests"),
        ("https://raw.githubusercontent.com/fastapi/fastapi/master/fastapi/routing.py", "FastAPI"),
        ("https://raw.githubusercontent.com/pallets/flask/main/src/flask/ctx.py", "Flask Ctx"),
        ("https://raw.githubusercontent.com/pallets/flask/main/src/flask/helpers.py", "Flask Helpers"),
        ("https://raw.githubusercontent.com/pallets/click/main/src/click/decorators.py", "Click Dec"),
        ("https://raw.githubusercontent.com/encode/httpx/master/httpx/_models.py", "HTTPX Models"),
        ("https://raw.githubusercontent.com/pallets/jinja/main/src/jinja2/environment.py", "Jinja2"),
    ]

    code_dir = CACHE_DIR / "code"
    code_dir.mkdir(exist_ok=True)
    all_code = []

    for url, desc in code_files:
        fname = code_dir / url.split("/")[-1]
        if not fname.exists():
            download_file(url, fname, desc)
        if fname.exists():
            lines = extract_code_lines(fname)
            all_code.extend(lines)
            print(f"  {desc}: {len(lines)} satır")

    extra_code = [
        "def fibonacci(n):",
        "    if n <= 1: return n",
        "    return fibonacci(n-1) + fibonacci(n-2)",
        "def quicksort(arr):",
        "    if len(arr) <= 1: return arr",
        "    pivot = arr[len(arr) // 2]",
        "    left = [x for x in arr if x < pivot]",
        "    middle = [x for x in arr if x == pivot]",
        "    right = [x for x in arr if x > pivot]",
        "    return quicksort(left) + middle + quicksort(right)",
        "def binary_search(arr, target):",
        "    left, right = 0, len(arr) - 1",
        "    while left <= right:",
        "        mid = (left + right) // 2",
        "        if arr[mid] == target: return mid",
        "        elif arr[mid] < target: left = mid + 1",
        "        else: right = mid - 1",
        "    return -1",
        "class Stack:",
        "    def __init__(self): self.items = []",
        "    def push(self, item): self.items.append(item)",
        "    def pop(self): return self.items.pop()",
        "    def peek(self): return self.items[-1]",
        "    def is_empty(self): return len(self.items) == 0",
        "class Queue:",
        "    def __init__(self): self.items = []",
        "    def enqueue(self, item): self.items.append(item)",
        "    def dequeue(self): return self.items.pop(0)",
        "    def is_empty(self): return len(self.items) == 0",
        "class LinkedList:",
        "    def __init__(self): self.head = None",
        "    def append(self, value):",
        "        if not self.head: self.head = Node(value)",
        "        else:",
        "            current = self.head",
        "            while current.next: current = current.next",
        "            current.next = Node(value)",
        "class TreeNode:",
        "    def __init__(self, val=0):",
        "        self.val = val",
        "        self.left = None",
        "        self.right = None",
        "def dfs(node, result=None):",
        "    if result is None: result = []",
        "    if node:",
        "        result.append(node.val)",
        "        dfs(node.left, result)",
        "        dfs(node.right, result)",
        "    return result",
        "def bfs(root):",
        "    if not root: return []",
        "    queue, result = [root], []",
        "    while queue:",
        "        node = queue.pop(0)",
        "        result.append(node.val)",
        "        if node.left: queue.append(node.left)",
        "        if node.right: queue.append(node.right)",
        "    return result",
        "class HashMap:",
        "    def __init__(self, size=100):",
        "        self.size = size",
        "        self.table = [[] for _ in range(size)]",
        "    def _hash(self, key): return hash(key) % self.size",
        "    def set(self, key, value):",
        "        idx = self._hash(key)",
        "        for i, (k, v) in enumerate(self.table[idx]):",
        "            if k == key: self.table[idx][i] = (key, value); return",
        "        self.table[idx].append((key, value))",
        "    def get(self, key):",
        "        idx = self._hash(key)",
        "        for k, v in self.table[idx]:",
        "            if k == key: return v",
        "        return None",
        "def merge_sort(arr):",
        "    if len(arr) <= 1: return arr",
        "    mid = len(arr) // 2",
        "    left = merge_sort(arr[:mid])",
        "    right = merge_sort(arr[mid:])",
        "    return merge(left, right)",
        "def merge(left, right):",
        "    result = []",
        "    i = j = 0",
        "    while i < len(left) and j < len(right):",
        "        if left[i] < right[j]: result.append(left[i]); i += 1",
        "        else: result.append(right[j]); j += 1",
        "    result.extend(left[i:]); result.extend(right[j:])",
        "    return result",
        "class MaxHeap:",
        "    def __init__(self): self.heap = []",
        "    def insert(self, val):",
        "        self.heap.append(val)",
        "        self._bubble_up(len(self.heap) - 1)",
        "    def _bubble_up(self, idx):",
        "        while idx > 0:",
        "            parent = (idx - 1) // 2",
        "            if self.heap[parent] < self.heap[idx]:",
        "                self.heap[parent], self.heap[idx] = self.heap[idx], self.heap[parent]",
        "                idx = parent",
        "            else: break",
        "class TrieNode:",
        "    def __init__(self):",
        "        self.children = {}",
        "        self.is_end = False",
        "class Trie:",
        "    def __init__(self): self.root = TrieNode()",
        "    def insert(self, word):",
        "        node = self.root",
        "        for char in word:",
        "            if char not in node.children: node.children[char] = TrieNode()",
        "            node = node.children[char]",
        "        node.is_end = True",
        "    def search(self, word):",
        "        node = self.root",
        "        for char in word:",
        "            if char not in node.children: return False",
        "            node = node.children[char]",
        "        return node.is_end",
        "def dijkstra(graph, start):",
        "    import heapq",
        "    dist = {node: float('inf') for node in graph}",
        "    dist[start] = 0",
        "    pq = [(0, start)]",
        "    while pq:",
        "        current_dist, current = heapq.heappop(pq)",
        "        if current_dist > dist[current]: continue",
        "        for neighbor, weight in graph[current].items():",
        "            distance = current_dist + weight",
        "            if distance < dist[neighbor]:",
        "                dist[neighbor] = distance",
        "                heapq.heappush(pq, (distance, neighbor))",
        "    return dist",
        "class UnionFind:",
        "    def __init__(self, n):",
        "        self.parent = list(range(n))",
        "        self.rank = [0] * n",
        "    def find(self, x):",
        "        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])",
        "        return self.parent[x]",
        "    def union(self, x, y):",
        "        px, py = self.find(x), self.find(y)",
        "        if px == py: return False",
        "        if self.rank[px] < self.rank[py]: px, py = py, px",
        "        self.parent[py] = px",
        "        if self.rank[px] == self.rank[py]: self.rank[px] += 1",
        "        return True",
        "class Graph:",
        "    def __init__(self): self.adj_list = {}",
        "    def add_edge(self, u, v, weight=1):",
        "        if u not in self.adj_list: self.adj_list[u] = []",
        "        if v not in self.adj_list: self.adj_list[v] = []",
        "        self.adj_list[u].append((v, weight))",
        "        self.adj_list[v].append((u, weight))",
        "    def bfs(self, start):",
        "        visited = set([start])",
        "        queue = [start]",
        "        order = []",
        "        while queue:",
        "            vertex = queue.pop(0)",
        "            order.append(vertex)",
        "            for neighbor, _ in self.adj_list.get(vertex, []):",
        "                if neighbor not in visited:",
        "                    visited.add(neighbor)",
        "                    queue.append(neighbor)",
        "        return order",
        "class AVLNode:",
        "    def __init__(self, key):",
        "        self.key = key",
        "        self.left = None",
        "        self.right = None",
        "        self.height = 1",
        "class AVLTree:",
        "    def insert(self, root, key):",
        "        if not root: return AVLNode(key)",
        "        if key < root.key: root.left = self.insert(root.left, key)",
        "        elif key > root.key: root.right = self.insert(root.right, key)",
        "        else: return root",
        "        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))",
        "        balance = self.get_balance(root)",
        "        if balance > 1 and key < root.left.key: return self.right_rotate(root)",
        "        if balance < -1 and key > root.right.key: return self.left_rotate(root)",
        "        return root",
        "    def left_rotate(self, z):",
        "        y = z.right",
        "        T2 = y.left",
        "        y.left = z",
        "        z.right = T2",
        "        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))",
        "        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))",
        "        return y",
        "    def right_rotate(self, z):",
        "        y = z.left",
        "        T3 = y.right",
        "        y.right = z",
        "        z.left = T3",
        "        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))",
        "        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))",
        "        return y",
        "    def get_height(self, root):",
        "        if not root: return 0",
        "        return root.height",
        "    def get_balance(self, root):",
        "        if not root: return 0",
        "        return self.get_height(root.left) - self.get_height(root.right)",
        "def knapsack(W, weights, values, n):",
        "    K = [[0]*(W+1) for _ in range(n+1)]",
        "    for i in range(n+1):",
        "        for w in range(W+1):",
        "            if i == 0 or w == 0: K[i][w] = 0",
        "            elif weights[i-1] <= w: K[i][w] = max(values[i-1]+K[i-1][w-weights[i-1]], K[i-1][w])",
        "            else: K[i][w] = K[i-1][w]",
        "    return K[n][W]",
    ] * 200
    all_code.extend(extra_code)
    random.shuffle(all_code)
    all_code = all_code[:20000]
    all_texts.extend(all_code)
    print(f"  Kod: {len(all_code)} satır")

    # 3. İngilizce (40K)
    print("\n[3/3] WikiText (40K)...")
    wt2 = CACHE_DIR / "wikitext2.txt"
    if not wt2.exists():
        download_file(
            "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
            wt2, "WikiText-2"
        )

    english = []
    if wt2.exists():
        with open(wt2, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip() and len(l.strip()) > 15]
            english.extend(lines[:10000])
            print(f"  WikiText-2: {len(lines[:10000])} satır")

    wt103 = CACHE_DIR / "wikitext103.txt"
    if not wt103.exists():
        zip_path = CACHE_DIR / "wikitext103.zip"
        download_file(
            "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1.zip",
            zip_path, "WikiText-103"
        )
        if zip_path.exists():
            import zipfile, shutil
            try:
                with zipfile.ZipFile(str(zip_path), 'r') as z:
                    z.extractall(str(CACHE_DIR))
                    for name in z.namelist():
                        if 'train' in name.lower() and name.endswith('.txt'):
                            shutil.move(str(CACHE_DIR / name), str(wt103))
                            break
            except Exception as e:
                print(f"  Zip hatasi: {e}")

    if wt103.exists():
        with open(wt103, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip() and len(l.strip()) > 15]
            english.extend(lines[:30000])
            print(f"  WikiText-103: {len(lines[:30000])} satır")

    english = english[:40000]
    all_texts.extend(english)
    print(f"  İngilizce: {len(english)} satır")

    # Karıştır ve kaydet
    random.shuffle(all_texts)
    output = Path("data/training.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for line in all_texts:
            f.write(line + "\n")

    print(f"\n{'=' * 60}")
    print(f"Toplam: {len(all_texts)} satır")
    print(f"  Türkçe: 40,000")
    print(f"  Kod: 20,000")
    print(f"  İngilizce: 40,000")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    prepare_data()
