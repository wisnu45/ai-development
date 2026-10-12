# AI Chatbot MVP --- Website Only

> **Dokumen utama pengembangan (single source of truth) untuk MVP AI
> Chatbot berbasis website.**\
> Cakupan dokumen ini meliputi persiapan development lokal, arsitektur
> aplikasi, implementasi fitur, Agentic RAG, orkestrasi n8n, keamanan,
> containerization, CI/CD, deployment VPS, monitoring, backup,
> pengujian, dan jalur scaling ke OpenShift AI.

------------------------------------------------------------------------

## Daftar Isi

-   [1. Tujuan dan Ruang Lingkup](#1-tujuan-dan-ruang-lingkup)
-   [2. Prinsip Arsitektur dan Keputusan
    Teknologi](#2-prinsip-arsitektur-dan-keputusan-teknologi)
-   [3. Arsitektur Sistem](#3-arsitektur-sistem)
-   [4. Alur Utama Sistem](#4-alur-utama-sistem)
-   [5. Prasyarat Development](#5-prasyarat-development)
-   [6. Struktur Repository](#6-struktur-repository)
-   [7. Persiapan Environment Lokal](#7-persiapan-environment-lokal)
-   [8. Konfigurasi Environment](#8-konfigurasi-environment)
-   [9. Menjalankan Infrastruktur Lokal dengan Docker
    Compose](#9-menjalankan-infrastruktur-lokal-dengan-docker-compose)
-   [10. Pengembangan Frontend
    Website](#10-pengembangan-frontend-website)
-   [11. Pengembangan Backend FastAPI](#11-pengembangan-backend-fastapi)
-   [12. Database PostgreSQL dan
    PGVector](#12-database-postgresql-dan-pgvector)
-   [13. Pipeline Dokumen dan
    PaddleOCR](#13-pipeline-dokumen-dan-paddleocr)
-   [14. Embedding dan Retrieval](#14-embedding-dan-retrieval)
-   [15. Implementasi Agentic RAG](#15-implementasi-agentic-rag)
-   [16. Integrasi LLM](#16-integrasi-llm)
-   [17. Integrasi n8n](#17-integrasi-n8n)
-   [18. Autentikasi, Otorisasi, dan Keamanan
    Data](#18-autentikasi-otorisasi-dan-keamanan-data)
-   [19. File Storage dan Enkripsi](#19-file-storage-dan-enkripsi)
-   [20. Pengujian dan Evaluasi](#20-pengujian-dan-evaluasi)
-   [21. Docker Containerization](#21-docker-containerization)
-   [22. GitHub dan Strategi Branch](#22-github-dan-strategi-branch)
-   [23. CI/CD dengan GitHub Actions](#23-cicd-dengan-github-actions)
-   [24. Deployment ke VPS Ubuntu](#24-deployment-ke-vps-ubuntu)
-   [25. Nginx, Domain, dan HTTPS](#25-nginx-domain-dan-https)
-   [26. Firewall, SSH Key, dan Hardening
    Server](#26-firewall-ssh-key-dan-hardening-server)
-   [27. Resource Limiting](#27-resource-limiting)
-   [28. Monitoring, Logging, Alerting, dan
    Backup](#28-monitoring-logging-alerting-dan-backup)
-   [29. Rencana Implementasi
    Bertahap](#29-rencana-implementasi-bertahap)
-   [30. Definition of Done](#30-definition-of-done)
-   [31. Troubleshooting](#31-troubleshooting)
-   [32. OpenShift AI --- Future Enterprise
    Scaling](#32-openshift-ai--future-enterprise-scaling)
-   [33. Checklist Akhir](#33-checklist-akhir)
-   [34. Referensi Dokumentasi](#34-referensi-dokumentasi)

------------------------------------------------------------------------

## 1. Tujuan dan Ruang Lingkup

Membangun MVP AI Chatbot yang diakses melalui **website**, memungkinkan
pengguna masuk ke sistem, mengunggah dokumen, mengajukan pertanyaan,
memperoleh jawaban yang berlandaskan dokumen yang diizinkan, dan melihat
riwayat percakapan.

### 1.1 Fitur MVP

-   Website chat responsif.
-   Login dan manajemen sesi pengguna.
-   Role dasar, misalnya `admin` dan `user`.
-   Upload dokumen yang diizinkan.
-   Pemrosesan dokumen melalui ekstraksi teks/OCR, normalisasi, dan
    chunking.
-   Pembuatan embedding dan penyimpanan vektor di PostgreSQL + PGVector.
-   Retrieval dokumen yang relevan ketika pengguna bertanya.
-   Jawaban LLM yang menggunakan konteks hasil retrieval.
-   Sitasi/referensi sumber dokumen pada jawaban jika sumber tersedia.
-   Riwayat percakapan.
-   Dashboard minimal untuk mengelola dokumen dan melihat status
    pemrosesan.
-   Workflow AI Agent dan otomasi melalui n8n.
-   Deployment berbasis Docker ke VPS Ubuntu.
-   HTTPS, pembatasan akses, logging, monitoring, alerting, dan backup.

### 1.2 Di luar cakupan MVP

-   **Mobile app native Flutter**: tidak dikembangkan pada MVP ini.
-   Training foundation model dari nol.
-   Multi-region deployment dan high availability kompleks.
-   Kubernetes/OpenShift sebagai platform produksi MVP.
-   OpenShift AI sebagai dependency wajib: hanya rencana scaling masa
    depan.
-   Agent yang dapat menjalankan tindakan berisiko tinggi tanpa
    persetujuan manusia.

### 1.3 Asumsi yang perlu dikonfirmasi saat implementasi

Detail yang belum ditetapkan tidak boleh dianggap sebagai keputusan
final. Tentukan dan dokumentasikan sebelum production:

-   Provider LLM: API hosted atau model lokal.
-   Model embedding dan dimensi vektornya.
-   Jenis, ukuran maksimum, dan jumlah dokumen yang diizinkan.
-   Domain dan subdomain.
-   Kebutuhan retensi data dan kebijakan penghapusan.
-   Kapasitas VPS aktual, kebutuhan RAM/CPU, dan perkiraan volume
    pengguna.
-   Mekanisme login: email/password, SSO, atau provider identitas lain.
-   Apakah n8n hanya untuk workflow internal atau juga workflow yang
    dipicu pengguna.
-   Target SLA dan jadwal backup.

------------------------------------------------------------------------

## 2. Prinsip Arsitektur dan Keputusan Teknologi

  ----------------------------------------------------------------------
  Lapisan                Teknologi              Tanggung jawab
  ---------------------- ---------------------- ------------------------
  Frontend               Next.js                UI chat, login, upload,
                                                riwayat, dashboard

  Backend/API            FastAPI                API, validasi,
                                                autentikasi, business
                                                logic, orkestrasi
                                                request

  Workflow automation    n8n                    Trigger, workflow AI
                                                Agent, conditional
                                                logic, notifikasi,
                                                integrasi layanan

  RAG                    Modul Agentic RAG di   Retrieval, penyusunan
                         backend                konteks, penggunaan
                                                tools, pembuatan jawaban

  LLM                    OpenAI API atau local  Generasi jawaban dan
                         LLM                    fungsi bahasa

  Embedding              Embedding model yang   Mengubah potongan
                         dipilih                dokumen dan pertanyaan
                                                menjadi vektor

  OCR                    PaddleOCR              OCR untuk dokumen/gambar
                                                hasil scan

  Database               PostgreSQL + PGVector  Pengguna, dokumen,
                                                chunk, embedding,
                                                metadata, riwayat chat

  File storage           Volume terproteksi     File asli dan hasil
                         atau                   pemrosesan
                         MinIO/S3-compatible    

  Enkripsi               AES-256 melalui        Enkripsi file/data
                         library yang           sensitif saat tersimpan
                         terpelihara            

  Container              Docker + Docker        Lingkungan konsisten
                         Compose                untuk lokal dan VPS

  Source control         GitHub                 Repository, pull
                                                request, code review

  CI                     GitHub Actions         Lint, test, security
                                                check, build image

  Image registry         GitHub Container       Penyimpanan Docker image
                         Registry (GHCR) atau   
                         registry yang          
                         disepakati             

  Reverse proxy          Nginx                  Routing HTTP(S), TLS
                                                termination, proxy ke
                                                layanan internal

  Infrastruktur          VPS Ubuntu             Menjalankan container
                                                produksi

  Security perimeter     UFW, SSH key, TLS      Pembatasan akses dan
                                                komunikasi terenkripsi

  Monitoring             Prometheus, Grafana,   Metrik
                         Uptime Kuma            infrastruktur/aplikasi
                                                dan uptime

  Logging                Loki atau solusi log   Pengumpulan dan
                         terpilih               pencarian log

  Alerting               Alertmanager           Notifikasi alert
                                                berbasis aturan

  Backup                 Backup database dan    Pemulihan ketika data
                         file terjadwal         hilang/rusak

  Future MLOps           OpenShift AI           Opsi enterprise untuk
                                                model lifecycle dan
                                                scaling
  ----------------------------------------------------------------------

### 2.1 Catatan arsitektur penting

-   FastAPI adalah pintu masuk API aplikasi; n8n tidak menggantikan
    backend.
-   Logika retrieval dan kontrol keamanan RAG harus berada di layanan
    yang dapat diuji dan diaudit. n8n dapat mengorkestrasi workflow,
    bukan menjadi satu-satunya lapisan keamanan.
-   LLM tidak boleh mendapat akses database atau filesystem tanpa
    pembatasan melalui tool/backend yang tervalidasi.
-   Gunakan jaringan internal Docker untuk layanan yang tidak perlu
    diakses publik.
-   Hanya port web yang diperlukan yang dibuka ke internet; port
    database, n8n, monitoring, dan service internal tidak boleh diekspos
    secara langsung.
-   Pemilihan model lokal memerlukan evaluasi RAM, CPU, GPU, throughput,
    lisensi, dan kualitas bahasa. Jangan menganggap VPS biasa cukup
    untuk menjalankan LLM besar.
-   OpenShift AI tidak perlu diinstal pada VPS MVP.

------------------------------------------------------------------------

## 3. Arsitektur Sistem

``` mermaid
flowchart LR
    U[Pengguna Website] --> HTTPS[Internet / HTTPS]
    HTTPS --> NGINX[Nginx Reverse Proxy]
    NGINX --> WEB[Next.js Frontend]
    WEB --> API[FastAPI Backend / API Gateway]

    API --> AUTH[Authentication & Authorization]
    API --> N8N[n8n Workflow Orchestration]
    API --> RAG[Agentic RAG Service]
    N8N <--> RAG

    RAG --> RET[Retrieval & Context Builder]
    RAG --> LLM[LLM Service]
    RAG --> EMB[Embedding Service]
    API --> OCR[PaddleOCR / Document Processing]

    RET --> PG[(PostgreSQL + PGVector)]
    API --> PG
    OCR --> FILES[Encrypted File Storage / MinIO]
    API --> FILES
    EMB --> PG
    LLM --> RAG

    API --> OBS[Logs / Metrics / Audit]
    NGINX --> OBS
    N8N --> OBS
```

### 3.1 Alur deployment dan DevOps

``` mermaid
flowchart LR
    DEV[Developer] --> GIT[GitHub Repository]
    GIT --> PR[Pull Request & Review]
    PR --> CI[GitHub Actions]
    CI --> TEST[Lint / Test / Security Scan]
    TEST --> BUILD[Build Docker Images]
    BUILD --> REG[Container Registry]
    REG --> DEP[Deploy ke VPS Ubuntu]
    DEP --> COMPOSE[Docker Compose]
    COMPOSE --> NGINX[Nginx + HTTPS]
    COMPOSE --> APP[Next.js / FastAPI / n8n]
    COMPOSE --> AI[LLM / OCR / RAG]
    COMPOSE --> DATA[PostgreSQL + PGVector / Storage]
    COMPOSE --> MON[Monitoring / Logging]
    DATA --> BACKUP[Backup Terjadwal]
```

### 3.2 Batas jaringan yang direkomendasikan

-   **Publik:** Nginx pada port 80 dan 443; SSH dibatasi ke alamat IP
    admin jika memungkinkan.
-   **Internal Docker:** frontend, backend, n8n, RAG/AI services,
    database, file storage, monitoring.
-   **Jangan publikasikan:** port PostgreSQL, dashboard n8n, Prometheus,
    Grafana, Loki, atau service internal tanpa kontrol akses tambahan.
-   Endpoint webhook n8n yang memang harus menerima request eksternal
    harus dibatasi, diautentikasi, dan dilindungi rate limit.

------------------------------------------------------------------------

## 4. Alur Utama Sistem

### 4.1 Alur upload dan ingest dokumen

1.  Pengguna login ke website.
2.  Frontend mengirim file ke endpoint upload backend.
3.  Backend memvalidasi identitas, role, tipe file, ukuran, dan kuota.
4.  File disimpan di lokasi terproteksi dan diberi ID unik.
5.  Backend membuat status pemrosesan, misalnya `uploaded` atau
    `queued`.
6.  Pipeline memeriksa apakah PDF memiliki teks yang dapat diekstrak
    atau membutuhkan OCR.
7.  PaddleOCR digunakan untuk halaman/gambar hasil scan bila diperlukan.
8.  Teks dibersihkan dan dinormalisasi.
9.  Teks dipecah menjadi chunk dengan metadata halaman dan sumber.
10. Embedding dibuat menggunakan model yang ditetapkan.
11. Chunk, embedding, dan metadata disimpan di PostgreSQL + PGVector.
12. Status berubah menjadi `completed`; kegagalan dicatat dan dapat
    dicoba ulang dengan batas tertentu.
13. Dashboard menampilkan status ingest.

### 4.2 Alur chat dengan RAG

1.  Pengguna mengirim pertanyaan dari website.
2.  Backend memvalidasi sesi, hak akses, batas ukuran, dan rate limit.
3.  Pertanyaan dinormalisasi dan diubah menjadi embedding.
4.  Retrieval mencari chunk relevan di PGVector dengan filter akses
    dokumen.
5.  Sistem memeriksa relevansi dan menyusun konteks dengan batas token.
6.  Agentic RAG menentukan apakah perlu retrieval ulang atau tool yang
    diizinkan.
7.  Backend mengirim prompt, konteks, dan instruksi ke LLM.
8.  Jawaban diperiksa dan dikaitkan dengan sumber yang mendukung bila
    tersedia.
9.  Backend menyimpan pesan, referensi sumber, dan metadata yang
    diperlukan.
10. Frontend menampilkan jawaban dan referensi dokumen.

### 4.3 Alur workflow n8n

1.  Trigger yang sah memulai workflow.
2.  Workflow memvalidasi input dan autentikasi.
3.  n8n memanggil endpoint backend atau layanan AI yang diizinkan.
4.  Node kondisi memutuskan jalur proses.
5.  Hasil ditulis ke sistem yang diizinkan atau notifikasi dikirim.
6.  Status, kegagalan, dan durasi workflow dicatat.
7.  Workflow menggunakan retry yang terbatas dan tidak mengulang
    tindakan non-idempotent tanpa pemeriksaan.

------------------------------------------------------------------------

## 5. Prasyarat Development

Gunakan versi stabil yang kompatibel dan kunci versi dependency setelah
baseline diuji.

### 5.1 Tools developer

-   Git.
-   Node.js versi LTS yang kompatibel dengan versi Next.js proyek.
-   Package manager Node: npm atau pnpm; pilih satu dan konsisten.
-   Python versi yang kompatibel dengan dependency FastAPI dan
    PaddleOCR.
-   `venv` untuk virtual environment Python.
-   Docker Engine/Desktop dan Docker Compose.
-   Editor kode seperti Visual Studio Code.
-   Browser modern.
-   Akun GitHub.
-   Kredensial provider LLM jika menggunakan API hosted.
-   Akses ke VPS Ubuntu dan domain ketika memasuki tahap deployment.

### 5.2 Verifikasi instalasi

``` bash
git --version
node --version
npm --version
python --version
docker --version
docker compose version
```

Jika perintah tidak ditemukan, perbaiki instalasi atau `PATH` sebelum
melanjutkan.

------------------------------------------------------------------------

## 6. Struktur Repository

Untuk MVP, gunakan monorepo agar deployment dan perubahan antarkomponen
mudah dilacak.

``` text
ai-chatbot-mvp/
├── apps/
│   └── web/                    # Next.js website
├── services/
│   └── api/                    # FastAPI backend dan API
├── ai/
│   ├── rag/                    # Retrieval, context builder, agent logic
│   ├── ingestion/              # Parsing, OCR, cleaning, chunking
│   └── embeddings/              # Adapter model embedding
├── n8n/
│   ├── workflows/              # Export workflow JSON yang terversi
│   └── README.md               # Dokumentasi workflow
├── infra/
│   ├── docker/                 # Dockerfile dan konfigurasi image
│   ├── nginx/                  # Reverse proxy dan konfigurasi TLS
│   ├── monitoring/             # Prometheus/Grafana/Loki/Alertmanager
│   ├── backup/                 # Script dan kebijakan backup
│   └── scripts/                # Deploy, migration, health check
├── tests/
│   ├── integration/
│   ├── e2e/
│   └── evaluation/             # Dataset evaluasi RAG
├── docs/
│   ├── architecture.md
│   ├── security.md
│   ├── operations.md
│   └── decisions/              # Architecture Decision Records (ADR)
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
├── .env.example
├── .gitignore
├── compose.yaml
├── Makefile                    # Opsional: shortcut command
└── README.md
```

Folder dapat disederhanakan pada fase awal, tetapi pisahkan tanggung
jawab frontend, backend, AI pipeline, workflow, dan infrastruktur.

------------------------------------------------------------------------

## 7. Persiapan Environment Lokal

### 7.1 Buat repository dan folder dasar

``` bash
mkdir ai-chatbot-mvp
cd ai-chatbot-mvp
git init

mkdir -p apps/web services/api
mkdir -p ai/rag ai/ingestion ai/embeddings
mkdir -p n8n/workflows infra/docker infra/nginx
mkdir -p infra/monitoring infra/backup infra/scripts
mkdir -p tests/integration tests/e2e tests/evaluation
mkdir -p docs/decisions .github/workflows
```

Buat repository GitHub, kemudian sambungkan:

``` bash
git remote add origin <URL_REPOSITORY_GITHUB>
git branch -M main
```

Ganti placeholder dengan URL repository milik proyek.

### 7.2 Buat frontend Next.js

Contoh menggunakan npm:

``` bash
npx create-next-app@latest apps/web
```

Pilih TypeScript, App Router, ESLint, dan opsi styling yang disepakati
tim. Catat pilihan dalam dokumentasi agar konsisten.

Jalankan frontend:

``` bash
cd apps/web
npm run dev
```

Buka `http://localhost:3000`, lalu kembali ke root repository sebelum
menjalankan perintah tingkat root.

### 7.3 Buat backend FastAPI

``` bash
cd services/api
python -m venv .venv
```

Aktifkan virtual environment:

**Windows PowerShell**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

``` bash
source .venv/bin/activate
```

Pasang dependency awal:

``` bash
python -m pip install --upgrade pip
pip install fastapi "uvicorn[standard]" pydantic-settings sqlalchemy \
  psycopg[binary] alembic python-multipart httpx
pip freeze > requirements.txt
```

Tambahkan dependency RAG, provider LLM, embedding, OCR, keamanan, dan
pengujian ketika modulnya mulai dibuat. Jangan memasang seluruh library
AI sekaligus sebelum kompatibilitas versi diuji.

Buat `services/api/main.py`:

``` python
from fastapi import FastAPI

app = FastAPI(title="AI Chatbot API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}
```

Jalankan:

``` bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Dokumentasi API development tersedia di `http://127.0.0.1:8000/docs`.
Jangan menggunakan konfigurasi development server sebagai konfigurasi
production.

### 7.4 Aturan file lokal

Tambahkan ke `.gitignore`:

``` gitignore
.env
.env.*
!.env.example
.venv/
__pycache__/
*.py[cod]
node_modules/
.next/
dist/
build/
coverage/
.pytest_cache/
.DS_Store
*.log
uploads/
data/
backups/
```

Sesuaikan pengecualian jika ada data contoh non-sensitif yang memang
sengaja dipelihara di repository. **Jangan commit secret, dokumen
privat, database dump, private key, atau file pengguna.**

------------------------------------------------------------------------

## 8. Konfigurasi Environment

Simpan contoh variabel pada `.env.example`; simpan nilai rahasia
sebenarnya di `.env` lokal atau secret manager. File `.env` tidak boleh
masuk Git.

Contoh variabel yang perlu dipetakan:

``` dotenv
# Application
APP_ENV=development
APP_NAME=ai-chatbot-mvp
APP_SECRET_KEY=replace-with-a-long-random-secret
FRONTEND_URL=http://localhost:3000
API_BASE_URL=http://localhost:8000

# Database
POSTGRES_DB=ai_chatbot
POSTGRES_USER=ai_chatbot_app
POSTGRES_PASSWORD=replace-with-a-strong-password
DATABASE_URL=postgresql+psycopg://ai_chatbot_app:replace-me@postgres:5432/ai_chatbot

# Authentication
JWT_SECRET_KEY=replace-with-a-different-long-random-secret
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# LLM
LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=choose-supported-model
LOCAL_LLM_BASE_URL=
LOCAL_LLM_MODEL=

# Embedding
EMBEDDING_PROVIDER=choose-provider
EMBEDDING_MODEL=choose-model
EMBEDDING_DIMENSION=replace-with-model-dimension

# OCR and ingestion
OCR_PROVIDER=paddleocr
MAX_UPLOAD_SIZE_MB=25
ALLOWED_UPLOAD_EXTENSIONS=pdf,png,jpg,jpeg
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_TOP_K=5

# n8n
N8N_ENCRYPTION_KEY=replace-with-a-long-random-secret
N8N_WEBHOOK_BASE_URL=http://n8n:5678
N8N_API_BASE_URL=http://n8n:5678

# Storage
STORAGE_BACKEND=local
UPLOAD_DIR=/app/data/uploads
ENCRYPTION_KEY=
S3_ENDPOINT=
S3_BUCKET=
S3_ACCESS_KEY=
S3_SECRET_KEY=

# Observability
LOG_LEVEL=INFO
OTEL_SERVICE_NAME=ai-chatbot-api
```

### 8.1 Ketentuan environment

-   Nama model, dimensi embedding, dan provider adalah konfigurasi
    contoh; tetapkan setelah memilih model yang benar.
-   `ENCRYPTION_KEY` harus dihasilkan dengan mekanisme kriptografi yang
    sesuai dan disimpan terpisah dari ciphertext.
-   Gunakan secret berbeda untuk JWT, enkripsi file, dan n8n.
-   Jangan memakai password contoh untuk lingkungan nyata.
-   Gunakan URL internal Docker untuk komunikasi antarkontainer di
    server.
-   Di production, konfigurasi frontend yang diekspos ke browser hanya
    boleh berisi nilai publik. Secret backend tidak boleh memakai
    prefiks environment publik Next.js.
-   Pastikan nilai secret tetap stabil setelah deployment, khususnya
    kunci enkripsi n8n agar credential yang tersimpan tidak menjadi
    tidak terbaca.

------------------------------------------------------------------------

## 9. Menjalankan Infrastruktur Lokal dengan Docker Compose

Gunakan Docker Compose untuk menyamakan dependency lintas developer.
Buat `compose.yaml` yang mendefinisikan layanan sesuai tahap
implementasi:

-   `web`: Next.js.
-   `api`: FastAPI.
-   `postgres`: image PostgreSQL yang menyediakan ekstensi PGVector atau
    image kompatibel yang telah diverifikasi.
-   `n8n`: workflow automation.
-   `minio` (opsional): object storage S3-compatible.
-   `llm` (opsional): hanya jika memilih local LLM dan perangkat
    memadai.
-   `nginx`: dapat ditambahkan pada tahap integrasi reverse proxy lokal.
-   Monitoring stack: tambahkan setelah metrik/log aplikasi tersedia.

### 9.1 Contoh Compose awal untuk database

Contoh potongan berikut adalah baseline database, bukan file Compose
lengkap seluruh aplikasi. Gunakan image PostgreSQL + PGVector yang
versinya telah diverifikasi dan dikunci.

``` yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend

volumes:
  postgres_data:

networks:
  backend:
    internal: true
```

Sebelum digunakan, pastikan image/tag tersedia dan kompatibel dengan
versi PostgreSQL yang dipilih. Jangan membuka port database ke publik.
Jika perlu akses dari host development, lakukan melalui konfigurasi
lokal terbatas, bukan konfigurasi production.

### 9.2 Urutan startup dan health check

1.  Mulai database dan tunggu status healthy.
2.  Jalankan migrasi schema.
3.  Mulai backend dan pastikan endpoint health merespons.
4.  Mulai frontend.
5.  Mulai n8n dan impor workflow yang diperlukan.
6.  Jalankan pengujian integrasi.
7.  Periksa log dan status semua container.

Perintah umum:

``` bash
docker compose config
docker compose up -d
docker compose ps
docker compose logs -f
docker compose down
```

Hindari `docker compose down -v` pada lingkungan yang memiliki data
penting karena opsi `-v` menghapus named volume terkait.

------------------------------------------------------------------------

## 10. Pengembangan Frontend Website

### 10.1 Halaman MVP

-   `/login`: autentikasi.
-   `/chat`: percakapan dengan AI.
-   `/history`: daftar dan detail riwayat percakapan.
-   `/documents`: upload dokumen dan status ingest.
-   `/dashboard`: ringkasan penggunaan dan status sistem untuk role yang
    berwenang.
-   `/admin/users`: opsional untuk administrasi pengguna.

Gunakan route protection pada frontend untuk pengalaman pengguna, tetapi
lakukan pemeriksaan otorisasi yang sebenarnya pada backend.

### 10.2 Komponen UI

-   `ChatLayout`
-   `ChatMessage`
-   `ChatInput`
-   `SourceCitation`
-   `ConversationList`
-   `DocumentUploader`
-   `DocumentStatusBadge`
-   `UserMenu`
-   `AdminDashboard`

### 10.3 Perilaku chat

-   Tampilkan status `sending`, `processing`, `success`, dan `error`.
-   Cegah pengiriman ganda secara tidak sengaja.
-   Batasi ukuran input dan tampilkan pesan validasi yang jelas.
-   Tampilkan sumber dokumen yang dipakai untuk jawaban.
-   Sediakan tindakan untuk membuat percakapan baru dan menghapus
    riwayat sesuai kebijakan.
-   Jangan menampilkan stack trace, prompt internal, secret, atau pesan
    error infrastruktur kepada pengguna.

### 10.4 Integrasi API

-   Gunakan satu konfigurasi base URL.
-   Kirim token sesi melalui mekanisme aman.
-   Tangani status `401`, `403`, `413`, `429`, dan `5xx`.
-   Jangan menyimpan token jangka panjang di tempat yang mudah diakses
    JavaScript jika dapat dihindari; pertimbangkan cookie `HttpOnly`,
    `Secure`, dan `SameSite`.
-   Terapkan loading state dan timeout yang wajar.

------------------------------------------------------------------------

## 11. Pengembangan Backend FastAPI

### 11.1 Pembagian modul

``` text
services/api/
├── main.py
├── app/
│   ├── api/
│   │   └── routes/
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── logging.py
│   ├── db/
│   │   ├── session.py
│   │   └── models/
│   ├── schemas/
│   ├── repositories/
│   ├── services/
│   │   ├── chat_service.py
│   │   ├── document_service.py
│   │   ├── rag_service.py
│   │   ├── llm_service.py
│   │   ├── embedding_service.py
│   │   └── storage_service.py
│   └── workers/
├── alembic/
├── tests/
├── requirements.txt
└── Dockerfile
```

### 11.2 Prinsip implementasi

-   Pisahkan router, schema request/response, business logic,
    repository, dan integrasi provider.
-   Validasi semua input dari klien dan webhook.
-   Gunakan parameter query atau ORM untuk mencegah SQL injection.
-   Terapkan timeout untuk panggilan LLM, OCR, embedding, storage, dan
    n8n.
-   Gunakan retry terbatas dengan exponential backoff hanya untuk
    kegagalan yang layak dicoba ulang.
-   Gunakan idempotency key atau status job untuk proses ingest agar
    file tidak diproses berkali-kali secara tidak sengaja.
-   Jangan menunggu proses OCR/embedding besar di request web sinkron
    tanpa batas waktu; gunakan background job/queue jika beban dan
    durasi proses membutuhkannya.
-   Terapkan pagination pada riwayat chat dan daftar dokumen.
-   Sediakan health check dan readiness check yang tidak membocorkan
    secret.

### 11.3 Endpoint API yang direncanakan

Endpoint berikut adalah rancangan awal; sesuaikan kontrak final saat
implementasi.

  ----------------------------------------------------------------------------
  Method              Endpoint                             Fungsi
  ------------------- ------------------------------------ -------------------
  `GET`               `/health`                            Liveness check

  `GET`               `/ready`                             Readiness check
                                                           dependency

  `POST`              `/api/v1/auth/login`                 Login

  `POST`              `/api/v1/auth/logout`                Logout/invalidate
                                                           session bila
                                                           didukung

  `GET`               `/api/v1/auth/me`                    Profil pengguna
                                                           aktif

  `POST`              `/api/v1/chat`                       Mengirim pertanyaan
                                                           dan mendapatkan
                                                           jawaban

  `GET`               `/api/v1/conversations`              Daftar percakapan

  `GET`               `/api/v1/conversations/{id}`         Detail percakapan

  `DELETE`            `/api/v1/conversations/{id}`         Hapus percakapan
                                                           sesuai kebijakan

  `POST`              `/api/v1/documents`                  Upload dokumen

  `GET`               `/api/v1/documents`                  Daftar dokumen yang
                                                           boleh diakses

  `GET`               `/api/v1/documents/{id}`             Detail/status
                                                           dokumen

  `DELETE`            `/api/v1/documents/{id}`             Hapus dokumen dan
                                                           data terkait

  `POST`              `/api/v1/documents/{id}/reprocess`   Proses ulang dengan
                                                           otorisasi

  `GET`               `/api/v1/admin/metrics`              Metrik admin yang
                                                           diotorisasi
  ----------------------------------------------------------------------------

### 11.4 Kontrak respons chat

Pertimbangkan respons terstruktur seperti berikut:

``` json
{
  "conversation_id": "uuid",
  "answer": "Jawaban berdasarkan konteks yang ditemukan.",
  "sources": [
    {
      "document_id": "uuid",
      "document_name": "panduan.pdf",
      "page": 4,
      "chunk_id": "uuid"
    }
  ],
  "status": "completed"
}
```

Jangan mengirim konten sumber yang pengguna tidak punya izin untuk
membacanya. Validasi ulang akses dokumen saat retrieval, bukan hanya
saat upload.

------------------------------------------------------------------------

## 12. Database PostgreSQL dan PGVector

### 12.1 Data utama

Schema minimum yang dapat digunakan sebagai titik awal:

-   `users`: identitas dan status pengguna.
-   `roles` / relasi role: role dan izin.
-   `documents`: pemilik, metadata, status ingest, lokasi file,
    checksum, dan waktu.
-   `document_chunks`: teks chunk, nomor halaman, urutan, metadata, dan
    referensi dokumen.
-   `document_embeddings`: vektor dan versi model embedding; dapat
    digabung ke tabel chunk jika sesuai.
-   `conversations`: pemilik percakapan, judul, dan timestamp.
-   `messages`: role pesan, konten, timestamp, dan metadata penggunaan.
-   `message_sources`: relasi jawaban dengan chunk/sumber.
-   `ingestion_jobs`: status, jumlah percobaan, error teredaksi, dan
    timestamp.
-   `audit_logs`: aktivitas sensitif seperti login, upload, delete,
    perubahan role, dan tindakan admin.

### 12.2 Migrasi dan ekstensi

Aktifkan PGVector melalui migrasi database:

``` sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Jalankan dengan role database yang memiliki izin yang sesuai. Jika
ekstensi tidak tersedia, pastikan image/server PostgreSQL yang digunakan
benar-benar menyertakan PGVector; jangan hanya mengulang perintah SQL.

Gunakan Alembic untuk versioning schema:

``` bash
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

Review migrasi yang dihasilkan sebelum dijalankan, khususnya pada
production.

### 12.3 Embedding dan index

-   Dimensi kolom vektor harus sama dengan dimensi model embedding yang
    dipilih.
-   Jangan mengganti model embedding tanpa strategi re-embedding atau
    versioning.
-   Mulai dari exact similarity search untuk baseline, lalu evaluasi
    kebutuhan index seperti HNSW atau IVFFlat sesuai volume dan pola
    query.
-   Pilih cosine distance, inner product, atau L2 berdasarkan
    karakteristik model dan normalisasi vektor.
-   Ukur relevansi retrieval sebelum mengoptimalkan performa.
-   Terapkan filter tenant/pemilik/izin pada query retrieval untuk
    mencegah kebocoran antar pengguna.

### 12.4 Backup dan data lifecycle

-   Backup database secara konsisten dan simpan di luar VPS yang sama.
-   Tentukan retensi, enkripsi backup, kontrol akses, dan jadwal
    penghapusan.
-   Uji restore secara berkala.
-   Ketika dokumen dihapus, hapus atau invalidasi file, chunk,
    embedding, referensi, dan cache yang terkait sesuai kebijakan
    retensi.

------------------------------------------------------------------------

## 13. Pipeline Dokumen dan PaddleOCR

### 13.1 Tahapan ingest

1.  Validasi format berdasarkan ekstensi dan konten file, bukan ekstensi
    saja.
2.  Terapkan batas ukuran, jumlah halaman, dan waktu pemrosesan.
3.  Simpan file secara terproteksi dengan nama internal yang tidak
    berasal langsung dari nama file pengguna.
4.  Jalankan pemeriksaan keamanan file yang sesuai dengan lingkungan.
5.  Ekstrak teks PDF bila tersedia.
6.  Gunakan PaddleOCR untuk halaman scan atau gambar yang memerlukan
    OCR.
7.  Normalisasi whitespace, pemisah halaman, dan karakter yang tidak
    diinginkan tanpa menghilangkan makna.
8.  Simpan metadata sumber, nomor halaman, pemilik, dan ID dokumen.
9.  Chunk teks berdasarkan batas yang ditentukan dan struktur dokumen.
10. Buat embedding dan simpan chunk beserta metadata.
11. Tandai job sukses atau gagal; sediakan retry dan pesan error yang
    tidak sensitif.

### 13.2 Format file

Tetapkan whitelist awal, misalnya PDF, PNG, dan JPEG. Dukungan DOCX atau
format lain harus ditambahkan setelah parser, keamanan, dan pengujiannya
siap.

### 13.3 Kualitas OCR

-   Uji dokumen teks dan scan dengan kualitas berbeda.
-   Evaluasi Bahasa Indonesia dan istilah domain.
-   Catat confidence OCR jika tersedia.
-   Simpan hasil OCR secara terproteksi.
-   Berikan status `needs_review` jika hasil ekstraksi tidak memadai
    untuk penggunaan yang membutuhkan akurasi tinggi.
-   Jangan menganggap hasil OCR selalu benar; jawaban RAG tetap harus
    dievaluasi terhadap sumber asli.

### 13.4 Ketahanan pipeline

-   Proses dokumen sebagai job yang memiliki status dan ID.
-   Hindari ingest duplikat dengan checksum atau mekanisme deduplikasi
    yang tepat.
-   Simpan jumlah percobaan dan kategori error.
-   Terapkan batas concurrency untuk mencegah OCR menghabiskan seluruh
    CPU/RAM VPS.
-   Pisahkan file asli dari hasil ekstraksi dan chunk hasil proses.

------------------------------------------------------------------------

## 14. Embedding dan Retrieval

### 14.1 Strategi embedding

Pilih model berdasarkan bahasa, panjang teks, kualitas retrieval,
lisensi, biaya, privasi, latency, dan resource. Catat nama model serta
versinya di konfigurasi dan metadata data.

### 14.2 Chunking

-   Mulai dengan ukuran chunk dan overlap yang dapat dikonfigurasi.
-   Pertahankan batas heading, paragraf, dan halaman jika memungkinkan.
-   Simpan metadata seperti `document_id`, `page_number`, `chunk_index`,
    `owner_id` atau scope akses, dan versi pipeline.
-   Hindari chunk terlalu besar yang memenuhi context window atau
    terlalu kecil yang kehilangan konteks.

### 14.3 Retrieval pipeline

1.  Terima pertanyaan yang telah divalidasi.
2.  Buat embedding query.
3.  Filter sumber berdasarkan hak akses pengguna.
4.  Cari kandidat terdekat di PGVector.
5.  Ambil top-k sesuai konfigurasi.
6.  Opsional: reranking bila evaluasi menunjukkan manfaat.
7.  Terapkan threshold/relevance check yang dievaluasi, bukan dipilih
    secara sembarang.
8.  Susun konteks dan sumber.
9.  Kembalikan jawaban dengan sitasi bila tersedia.

### 14.4 Perlindungan terhadap prompt injection dalam dokumen

Dokumen adalah input tidak tepercaya. Instruksi yang ditemukan di dalam
dokumen tidak boleh menggantikan system policy. Jangan biarkan teks
dokumen memerintahkan LLM untuk membuka secret, mengubah role, mengakses
file lain, atau menjalankan tindakan di luar tool yang diizinkan.

------------------------------------------------------------------------

## 15. Implementasi Agentic RAG

Agentic RAG menambahkan kemampuan memilih langkah atau tool yang
diizinkan di atas retrieval biasa. Untuk MVP, implementasikan dengan
ruang tindakan yang terbatas dan dapat diaudit.

### 15.1 Komponen

-   **Query handling:** normalisasi dan klasifikasi kebutuhan
    pertanyaan.
-   **Retriever:** mencari chunk yang sesuai di PGVector.
-   **Context builder:** membatasi token, mengurutkan sumber, dan
    mempertahankan metadata.
-   **LLM adapter:** interface ke provider model.
-   **Agent policy:** menentukan apakah retrieval ulang atau tool
    terbatas dibutuhkan.
-   **Tool registry:** hanya mendaftarkan tool yang eksplisit diizinkan.
-   **Answer validator:** memeriksa format dan keberadaan referensi
    sumber.
-   **Conversation memory:** menyimpan riwayat yang diperlukan; jangan
    menganggap riwayat sebagai sumber fakta yang selalu benar.
-   **Tracing/audit:** mencatat langkah dan hasil tanpa merekam secret
    atau data sensitif berlebihan.

### 15.2 Aturan agent

-   Retrieval dan jawaban harus tunduk pada hak akses pengguna.
-   Batasi jumlah iterasi agent, tool call, durasi, dan token.
-   Gunakan allowlist tool, parameter schema, dan validasi input.
-   Tool yang mengubah data atau melakukan aksi eksternal memerlukan
    otorisasi backend dan, jika berisiko, konfirmasi manusia.
-   Jangan memberi akses shell, SQL bebas, atau filesystem umum kepada
    agent.
-   Jika sumber tidak cukup, agent harus menyatakan keterbatasan
    alih-alih mengarang.
-   Jawaban yang bersumber dari dokumen harus menampilkan rujukan yang
    benar-benar digunakan.
-   Log alasan keputusan secara ringkas tanpa mengekspos
    chain-of-thought privat atau secret.

### 15.3 Evaluasi RAG

Buat dataset evaluasi yang berisi pertanyaan, dokumen relevan, jawaban
yang diharapkan, dan sumber yang benar. Ukur sekurang-kurangnya:

-   Retrieval relevance / hit rate.
-   Kebenaran sitasi.
-   Faithfulness jawaban terhadap konteks.
-   Kelengkapan jawaban.
-   Latency end-to-end.
-   Tingkat kegagalan dan fallback.
-   Perilaku saat jawaban tidak ada di dokumen.

Tetapkan target angka setelah baseline diukur. Jangan mengklaim kualitas
sistem hanya dari beberapa contoh manual.

------------------------------------------------------------------------

## 16. Integrasi LLM

### 16.1 Adapter provider

Gunakan abstraksi provider agar aplikasi tidak bergantung pada satu API
saja. Kontrak minimal mencakup:

-   `generate_answer(messages, context, options)`
-   Timeout dan pembatasan output.
-   Penanganan rate limit dan error provider.
-   Pemakaian token/biaya jika provider menyediakan.
-   Pencatatan model dan versi.
-   Redaksi informasi sensitif.

### 16.2 Hosted API atau local LLM

**Hosted API** - Lebih cepat untuk memulai jika konektivitas dan
kebijakan data memungkinkan. - Memerlukan API key, pengendalian biaya,
batas penggunaan, dan review kebijakan privasi/data provider.

**Local LLM** - Dapat memberikan kontrol lebih besar atas lokasi data. -
Memerlukan pengukuran RAM/CPU/GPU, penyimpanan model, latency,
concurrency, dan lisensi. - Tidak otomatis lebih murah atau lebih
cepat. - Bisa ditempatkan pada host atau layanan terpisah jika resource
VPS tidak mencukupi.

### 16.3 Batas biaya dan reliabilitas

-   Tetapkan batas input/output token dan timeout.
-   Batasi request per pengguna.
-   Gunakan retry terbatas untuk error sementara.
-   Jangan retry error validasi atau autentikasi.
-   Tampilkan fallback yang ramah saat provider tidak tersedia.
-   Simpan metrik penggunaan tanpa mencatat isi prompt sensitif secara
    default.

------------------------------------------------------------------------

## 17. Integrasi n8n

### 17.1 Peran n8n

n8n digunakan untuk workflow automation, misalnya:

-   Memicu pipeline ingest melalui endpoint backend yang tervalidasi.
-   Mengatur workflow AI Agent yang memakai LLM, OCR, atau RAG melalui
    API internal.
-   Conditional logic untuk mengarahkan hasil ke jalur yang tepat.
-   Mengirim notifikasi status atau kegagalan ke kanal yang disetujui.
-   Integrasi layanan pihak ketiga pada fase berikutnya.

### 17.2 Workflow MVP yang direkomendasikan

**Workflow A --- Document Processing** 1. Trigger internal/webhook
terautentikasi. 2. Validasi `document_id`. 3. Minta backend memproses
dokumen. 4. Periksa status job. 5. Jika sukses, perbarui status workflow
dan kirim notifikasi bila diperlukan. 6. Jika gagal, catat error yang
telah direduksi dan lakukan retry terbatas.

**Workflow B --- AI Assistant** 1. Terima event/request yang
tervalidasi. 2. Panggil endpoint Agentic RAG. 3. Evaluasi status dan
struktur hasil. 4. Kirim respons ke sistem pemanggil. 5. Catat durasi,
status, dan ID korelasi.

**Workflow C --- Operational Alert** 1. Terima alert dari monitoring. 2.
Terapkan kondisi dan deduplikasi. 3. Kirim notifikasi kepada pihak yang
berwenang. 4. Hindari menyertakan token, prompt, dokumen, atau data
pribadi pada pesan alert.

### 17.3 Keamanan n8n

-   Tetapkan `N8N_ENCRYPTION_KEY` yang kuat dan stabil.
-   Lindungi editor n8n di balik autentikasi dan akses jaringan
    terbatas.
-   Jangan membuka port editor ke internet tanpa kontrol yang memadai.
-   Gunakan credential store n8n dan secret environment yang tepat.
-   Batasi webhook dengan autentikasi, validasi input, dan rate limit.
-   Review workflow sebelum dipublikasikan.
-   Simpan export workflow JSON di Git tanpa credential atau data
    pengguna.
-   Terapkan timeout, retry, dan batas concurrency.
-   Batasi hak akses service account; jangan menggunakan database
    superuser.

### 17.4 Batas tanggung jawab

Backend tetap menjadi sumber otoritatif untuk autentikasi, otorisasi,
validasi, retrieval dengan filter akses, dan perubahan data. n8n
mengorkestrasi proses yang sudah diizinkan backend.

------------------------------------------------------------------------

## 18. Autentikasi, Otorisasi, dan Keamanan Data

### 18.1 Authentication

-   Gunakan mekanisme password hashing yang modern bila mengelola
    password sendiri.
-   Gunakan token/session dengan expiry, rotasi/revocation sesuai
    kebutuhan, dan secret yang kuat.
-   Pertimbangkan cookie `HttpOnly`, `Secure`, `SameSite` untuk sesi
    website.
-   Terapkan proteksi brute-force dan rate limit pada login.
-   Jangan mencatat password, token, API key, atau cookie ke log.

### 18.2 Authorization / RBAC

Contoh role minimum:

-   `admin`: mengelola pengguna dan resource sesuai kebijakan.
-   `user`: menggunakan chat dan mengakses dokumen/percakapan yang
    menjadi haknya.

Periksa izin di backend untuk setiap operasi. Menyembunyikan tombol pada
frontend bukan kontrol akses.

### 18.3 Keamanan API

-   CORS hanya mengizinkan origin yang diperlukan.
-   Validasi schema dan ukuran request.
-   Terapkan rate limit dan timeout.
-   Gunakan HTTPS untuk semua akses eksternal.
-   Validasi webhook dan gunakan secret/signature jika didukung.
-   Hindari SSRF: batasi URL yang dapat dipanggil backend dan jangan
    menerima target internal sembarang dari pengguna.
-   Gunakan error response yang tidak membocorkan stack trace.

### 18.4 Audit

Catat event penting: login berhasil/gagal, upload, download, delete,
perubahan role, perubahan konfigurasi, dan tindakan admin. Simpan hanya
informasi yang diperlukan, batasi akses log, dan tentukan retensi.

### 18.5 Data minimization

-   Hindari mengirim data sensitif ke provider LLM sebelum kebijakan
    privasi disetujui.
-   Redaksi secret dan data pribadi pada log.
-   Terapkan pemisahan data berdasarkan pemilik atau scope.
-   Tentukan prosedur penghapusan data dan backup yang sesuai kebijakan
    retensi.

------------------------------------------------------------------------

## 19. File Storage dan Enkripsi

### 19.1 Pilihan storage

-   **Volume lokal terproteksi:** sederhana untuk MVP skala kecil,
    tetapi perlu backup dan pengelolaan kapasitas.
-   **MinIO/S3-compatible:** berguna untuk object storage, lifecycle,
    dan pemisahan storage dari aplikasi; memerlukan credential dan
    kebijakan akses yang aman.

Pilih salah satu sebagai backend utama, dan abstraksikan operasi storage
agar pilihan dapat diganti.

### 19.2 AES-256

-   Gunakan library kriptografi terpelihara dan mode authenticated
    encryption seperti AES-256-GCM bila sesuai.
-   Jangan merancang algoritma enkripsi sendiri.
-   Gunakan nonce/IV yang unik sesuai persyaratan mode.
-   Simpan kunci terpisah dari file terenkripsi dan backup ciphertext.
-   Buat rencana rotasi kunci dan pemulihan.
-   Enkripsi file saat tersimpan; TLS melindungi komunikasi saat
    transit. Keduanya bukan pengganti satu sama lain.

### 19.3 Upload yang aman

-   Validasi tipe konten dan ukuran.
-   Gunakan nama file internal acak.
-   Simpan di luar direktori publik website.
-   Jangan mengeksekusi atau merender file upload secara langsung.
-   Batasi akses download melalui endpoint yang melakukan otorisasi.
-   Pertimbangkan antivirus/malware scanning sesuai kebutuhan risiko.
-   Pastikan penghapusan file dan metadata dilakukan konsisten.

------------------------------------------------------------------------

## 20. Pengujian dan Evaluasi

### 20.1 Jenis pengujian

-   **Unit test:** fungsi chunking, validasi, auth policy, prompt
    builder, adapter, dan storage.
-   **Integration test:** FastAPI + PostgreSQL/PGVector, OCR pipeline,
    LLM adapter, n8n webhook, dan storage.
-   **End-to-end test:** login, upload, ingest, chat, lihat sumber, dan
    riwayat.
-   **Security test:** otorisasi lintas pengguna, upload berbahaya, rate
    limit, secret leakage, SSRF, dan akses endpoint internal.
-   **Load test:** request chat bersamaan, ingest dokumen, dan
    penggunaan memori/CPU.
-   **RAG evaluation:** relevansi retrieval, faithfulness, citation
    correctness, dan fallback.

### 20.2 Skenario minimum

-   Pengguna yang belum login tidak dapat mengakses endpoint
    terlindungi.
-   Pengguna A tidak dapat membaca dokumen atau riwayat pengguna B.
-   File tidak didukung atau melebihi batas ditolak.
-   Dokumen scan dapat masuk pipeline OCR.
-   Kegagalan OCR tidak membuat status dokumen menjadi sukses.
-   Pertanyaan yang memiliki sumber menampilkan sumber yang benar.
-   Pertanyaan di luar pengetahuan dokumen tidak dijawab seolah-olah
    fakta pasti.
-   LLM provider timeout ditangani dengan error/fallback.
-   Workflow n8n tidak dapat dipanggil oleh pihak tidak berwenang.
-   Secret tidak muncul pada response, frontend bundle, atau log.
-   Database dan storage dapat dipulihkan dari backup yang diuji.

### 20.3 Gerbang rilis

Jangan rilis hanya karena halaman website dapat dibuka. Rilis harus
menunggu unit/integration test, pemeriksaan keamanan dasar, evaluasi
RAG, verifikasi backup, health check, dan rollback plan.

------------------------------------------------------------------------

## 21. Docker Containerization

### 21.1 Dockerfile

Buat Dockerfile terpisah untuk frontend dan backend. Gunakan multi-stage
build bila sesuai, image dasar tepercaya, dependency terkunci, dan user
non-root bila memungkinkan.

Contoh kerangka backend:

``` dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Ini adalah kerangka awal. Sesuaikan lokasi entry point, dependency
sistem OCR, health check, dan proses production. Uji build sebelum
dipakai. Jangan mengandalkan `--reload` di production.

### 21.2 Aturan container

-   Pin image versi; hindari tag `latest` untuk deployment yang perlu
    reproduktif.
-   Gunakan `.dockerignore`.
-   Jalankan proses sebagai non-root jika kompatibel.
-   Gunakan filesystem read-only bila memungkinkan dan mount volume
    hanya pada path yang diperlukan.
-   Jangan bake secret ke dalam image atau layer build.
-   Tambahkan health check.
-   Batasi capability dan privilege.
-   Tetapkan restart policy yang sesuai.
-   Pisahkan network publik dan internal.

### 21.3 Image yang dirilis

Tentukan image yang diperlukan berdasarkan implementasi aktual, minimal:

-   `web`
-   `api`
-   worker ingest jika dipisahkan
-   konfigurasi/service `n8n`
-   reverse proxy `nginx`
-   database image yang kompatibel dengan PGVector
-   monitoring stack jika diaktifkan

LLM lokal mungkin membutuhkan service/host terpisah; jangan memaksakan
seluruh komponen pada satu VPS tanpa uji resource.

------------------------------------------------------------------------

## 22. GitHub dan Strategi Branch

### 22.1 Branch

-   `main`: branch stabil yang siap dirilis.
-   `feature/<nama>`: fitur baru.
-   `fix/<nama>`: perbaikan bug.
-   `hotfix/<nama>`: perbaikan mendesak untuk production.

### 22.2 Pull request

Setiap PR harus memuat:

-   Ringkasan perubahan.
-   Dampak terhadap API, schema, security, dan deployment.
-   Bukti test.
-   Perubahan konfigurasi atau migrasi.
-   Catatan rollback bila perubahan berisiko.

Aktifkan branch protection untuk `main`, wajibkan review dan status CI
yang sesuai kemampuan repository.

### 22.3 Commit

Gunakan pesan commit yang informatif, misalnya:

``` text
feat(web): add document upload page
feat(rag): add pgvector retrieval
fix(auth): enforce document ownership
test(api): add chat integration tests
chore(devops): add health checks
```

Jangan commit `.env`, credential, database dump, private key, atau
dokumen internal.

------------------------------------------------------------------------

## 23. CI/CD dengan GitHub Actions

### 23.1 Continuous Integration

Workflow CI minimal:

1.  Trigger saat pull request dan push.
2.  Checkout kode.
3.  Setup Node.js dan Python.
4.  Install dependency dari lockfile.
5.  Jalankan lint dan type checking frontend.
6.  Jalankan formatter/linter dan unit test backend.
7.  Jalankan test integrasi yang tidak memerlukan secret production.
8.  Jalankan dependency/security scanning.
9.  Build Docker image.
10. Publikasikan status check agar PR dapat dinilai.

### 23.2 Continuous Deployment

1.  Deployment dipicu dari branch/tag yang disetujui.
2.  CI membangun image bertag commit SHA.
3.  Image dipindai dan dipush ke GHCR/registry.
4.  Workflow melakukan autentikasi ke VPS dengan SSH key khusus
    deployment atau mekanisme deployment yang lebih aman.
5.  VPS mengambil image berdasarkan tag/immutable digest.
6.  Jalankan backup pre-deploy jika perubahan database berisiko.
7.  Jalankan migrasi secara terkontrol.
8.  Terapkan Compose baru.
9.  Tunggu health check.
10. Jalankan smoke test.
11. Jika gagal, hentikan promosi rilis dan lakukan rollback sesuai
    prosedur.
12. Catat versi rilis dan hasil deployment.

### 23.3 Secrets GitHub

Simpan secret sebagai GitHub Actions Secrets/Environment Secrets,
misalnya:

-   `DEPLOY_HOST`
-   `DEPLOY_USER`
-   `DEPLOY_SSH_KEY`
-   `DEPLOY_KNOWN_HOSTS`
-   `REGISTRY_USERNAME` / token registry bila diperlukan

Gunakan token dengan scope minimum. Verifikasi host key SSH; jangan
menonaktifkan pemeriksaan host key untuk mempermudah deployment.
Pisahkan secret staging dan production.

### 23.4 Registry

GHCR adalah opsi default yang praktis karena terintegrasi dengan GitHub.
Atur visibility, akses pull di VPS, retensi image, dan kebijakan tag.
Jangan mengandalkan tag `latest` sebagai satu-satunya penanda versi.

------------------------------------------------------------------------

## 24. Deployment ke VPS Ubuntu

### 24.1 Persiapan server

-   Pilih VPS berdasarkan pengukuran beban dan kebutuhan OCR/LLM.
-   Gunakan Ubuntu LTS yang masih mendapat dukungan.
-   Perbarui paket keamanan.
-   Buat user administrator non-root.
-   Gunakan SSH key.
-   Aktifkan UFW.
-   Pasang Docker Engine dan plugin Docker Compose dari sumber resmi
    yang sesuai.
-   Buat direktori aplikasi dan konfigurasi dengan permission terbatas.
-   Siapkan DNS domain/subdomain.
-   Tentukan lokasi backup di luar server.

### 24.2 Direktori server

Contoh:

``` text
/opt/ai-chatbot/
├── compose.yaml
├── .env
├── nginx/
├── scripts/
└── release/
```

Pastikan `.env` hanya dapat dibaca user yang memerlukan. Direktori data
database, upload, n8n, dan monitoring harus memakai volume persisten
dengan permission yang benar.

### 24.3 Proses deploy

1.  Pastikan secret dan konfigurasi environment sudah disiapkan.
2.  Login ke registry menggunakan kredensial minimum.
3.  Pull image dengan tag release yang spesifik.
4.  Backup database dan data yang relevan.
5.  Validasi Compose dengan `docker compose config`.
6.  Jalankan migrasi database yang sudah direview.
7.  Terapkan container baru.
8.  Periksa status, health check, dan log.
9.  Jalankan smoke test melalui domain HTTPS.
10. Pantau error dan resource setelah rilis.
11. Jika gagal, rollback image/config sesuai runbook dan periksa
    kompatibilitas migrasi.

### 24.4 Persyaratan produksi

-   Tidak ada mode debug atau auto-reload.
-   Tidak ada secret default.
-   Database tidak diekspos ke internet.
-   Semua service kritis memiliki health check.
-   Log tidak menyimpan kredensial atau konten sensitif secara
    berlebihan.
-   Backup telah diuji restore.
-   Ada rencana rollback dan kontak penanggung jawab operasional.

------------------------------------------------------------------------

## 25. Nginx, Domain, dan HTTPS

### 25.1 Tugas Nginx

-   Menerima trafik publik.
-   Meneruskan request ke frontend/backend sesuai routing.
-   Menangani TLS.
-   Mengatur header keamanan, ukuran request, timeout, dan konfigurasi
    proxy.
-   Menghindari akses langsung ke port internal container.

### 25.2 Domain

Contoh pemisahan domain (sesuaikan kebutuhan):

-   `chat.example.com`: website.
-   `api.example.com`: API, atau gunakan path `/api` pada domain yang
    sama.
-   `n8n.example.com`: sebaiknya tidak dipublikasikan secara umum; jika
    diperlukan, batasi dengan VPN/allowlist dan autentikasi kuat.

Ganti `example.com` dengan domain sebenarnya. Jangan menganggap domain
ini sudah dikonfigurasi.

### 25.3 TLS

-   Gunakan sertifikat valid dari Let's Encrypt atau penyedia yang
    sesuai.
-   Otomatiskan renewal dan pantau kegagalannya.
-   Redirect HTTP ke HTTPS setelah sertifikat siap.
-   Terapkan konfigurasi TLS yang aman.
-   Gunakan cookie secure pada website.
-   Jangan menaruh private key sertifikat di repository.

### 25.4 WebSocket/streaming

Jika chat memakai streaming atau WebSocket, atur header upgrade,
buffering, dan timeout sesuai protokol yang dipakai. Uji perilaku
koneksi putus dan reconnect.

------------------------------------------------------------------------

## 26. Firewall, SSH Key, dan Hardening Server

### 26.1 UFW

Kebijakan minimum adalah menolak trafik masuk yang tidak diperlukan dan
hanya membuka port layanan publik. Sebelum mengubah aturan SSH, pastikan
akses SSH yang aktif tidak terputus.

Contoh kebijakan umum (sesuaikan port SSH dan kebijakan organisasi):

``` bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status verbose
```

Jika SSH menggunakan port khusus, izinkan port itu terlebih dahulu.
Jangan membuka port PostgreSQL, n8n, Prometheus, Grafana, atau port
aplikasi internal untuk publik tanpa kebutuhan yang terdokumentasi.

### 26.2 SSH key

-   Buat key pair di workstation administrator.
-   Pasang public key pada user server.
-   Uji login menggunakan key pada sesi baru sebelum menonaktifkan
    metode lama.
-   Lindungi private key dengan passphrase dan jangan membagikannya.
-   Nonaktifkan login root langsung dan password authentication setelah
    key access terverifikasi serta prosedur pemulihan tersedia.
-   Batasi sumber IP untuk SSH bila memungkinkan.
-   Gunakan key khusus deployment CI/CD, bukan private key pribadi.

### 26.3 Hardening tambahan

-   Terapkan update keamanan berkala.
-   Minimalkan paket dan service yang berjalan.
-   Lindungi Docker socket karena aksesnya setara hak tinggi pada host.
-   Batasi akses admin ke n8n dan monitoring.
-   Audit user, SSH key, firewall rule, dan credential secara berkala.
-   Simpan catatan perubahan konfigurasi server.

------------------------------------------------------------------------

## 27. Resource Limiting

Tetapkan limit berdasarkan pengukuran, bukan angka acak. Docker Compose
mendukung beberapa bentuk konfigurasi resource; verifikasi perilakunya
pada versi Docker Compose yang digunakan.

### 27.1 Komponen yang perlu dibatasi

-   Frontend Next.js.
-   Backend FastAPI.
-   n8n.
-   Worker OCR/ingestion.
-   LLM lokal jika digunakan.
-   PostgreSQL + PGVector.
-   Monitoring/logging.

### 27.2 Kebijakan

-   Tetapkan CPU dan memory limit/request yang sesuai untuk service yang
    didukung.
-   Batasi concurrency OCR dan embedding.
-   Tetapkan maksimum ukuran upload, halaman dokumen, token
    input/output, timeout, dan request bersamaan.
-   Pantau OOM kill, latency, swap, disk, koneksi database, dan panjang
    antrean job.
-   Sediakan kapasitas disk untuk database, file, image Docker, log, dan
    backup.
-   Jangan menjalankan local LLM besar pada VPS sebelum benchmark
    membuktikan resource mencukupi.

### 27.3 Penyesuaian kapasitas

Lakukan load test dan catat baseline CPU, RAM, latency, throughput,
serta error. Tingkatkan resource atau pisahkan layanan ke host lain
berdasarkan data tersebut. Resource limit yang terlalu rendah juga dapat
menyebabkan crash atau timeout.

------------------------------------------------------------------------

## 28. Monitoring, Logging, Alerting, dan Backup

### 28.1 Monitoring

Opsi komponen:

-   **Prometheus:** mengumpulkan metrik.
-   **Grafana:** dashboard.
-   **Uptime Kuma:** pemeriksaan ketersediaan endpoint.
-   **Alertmanager:** mengelola dan meneruskan alert.
-   Exporter/metrics endpoint: gunakan sesuai kebutuhan host, Docker,
    database, dan aplikasi.

Pantau sekurang-kurangnya:

-   Uptime website dan API.
-   CPU, RAM, disk, dan restart container.
-   Latency API dan error rate.
-   Kegagalan login yang mencurigakan dan rate limit.
-   Durasi OCR/ingest dan jumlah job gagal.
-   Latency LLM, timeout, token/cost jika tersedia.
-   Kualitas/keberhasilan retrieval dan jumlah jawaban tanpa sumber.
-   Koneksi dan ukuran database.
-   Status backup dan umur backup terakhir.

### 28.2 Logging

Loki dapat digunakan untuk sentralisasi log; pilihan ini perlu
disesuaikan dengan kapasitas VPS.

-   Gunakan structured logging dan correlation/request ID.
-   Redaksi token, password, API key, dan data sensitif.
-   Tetapkan retensi dan rotasi log.
-   Pisahkan application log, access log, audit log, dan log
    operasional.
-   Jangan menulis isi dokumen atau prompt lengkap ke log secara
    default.

### 28.3 Alerting

Buat alert untuk:

-   Website/API tidak tersedia.
-   Error rate atau latency melewati ambang yang ditentukan.
-   Disk hampir penuh.
-   Memori tinggi atau container berulang kali restart.
-   Database tidak siap.
-   Pipeline ingest gagal berulang.
-   Provider LLM tidak tersedia atau rate limit meningkat.
-   Backup gagal atau terlalu lama tidak berjalan.
-   Sertifikat TLS mendekati kedaluwarsa.

Pilih ambang setelah mengamati baseline dan arahkan alert ke kanal yang
disetujui, misalnya email, Telegram, atau Slack.

### 28.4 Backup

Backup mencakup:

-   PostgreSQL: database, schema, dan data aplikasi.
-   File asli dan hasil OCR yang harus dipertahankan.
-   Konfigurasi deployment non-secret.
-   Export workflow n8n.
-   Credential/encryption key hanya melalui mekanisme backup secret yang
    sangat terbatas dan terenkripsi; jangan menyimpannya bersama backup
    biasa tanpa kontrol.

Kebijakan minimum:

1.  Tentukan jadwal, retensi, dan lokasi backup off-site.
2.  Enkripsi backup saat transit dan tersimpan.
3.  Pisahkan hak akses backup dari hak akses aplikasi.
4.  Monitor hasil backup dan kegagalan.
5.  Lakukan uji restore berkala pada lingkungan terisolasi.
6.  Dokumentasikan RPO/RTO yang disetujui.
7.  Pastikan backup dapat dipulihkan tanpa bergantung pada VPS sumber.

------------------------------------------------------------------------

## 29. Rencana Implementasi Bertahap

### Fase 0 --- Keputusan dan setup

-   [ ] Tetapkan scope, role, jenis dokumen, dan batas upload.
-   [ ] Putuskan hosted LLM atau local LLM.
-   [ ] Pilih model embedding dan catat dimensinya.
-   [ ] Putuskan storage backend dan kebijakan enkripsi.
-   [ ] Buat repository GitHub dan struktur folder.
-   [ ] Setup Node, Python, Docker, dan environment lokal.
-   [ ] Buat `.env.example` dan `.gitignore`.
-   [ ] Buat ADR untuk keputusan arsitektur penting.

**Output:** repository dasar dan keputusan teknologi terdokumentasi.

### Fase 1 --- Fondasi aplikasi

-   [ ] Buat Next.js website.
-   [ ] Buat FastAPI dengan health endpoint.
-   [ ] Buat konfigurasi environment dan logging.
-   [ ] Setup PostgreSQL + PGVector.
-   [ ] Setup migrasi Alembic.
-   [ ] Setup Docker Compose development.
-   [ ] Tambahkan lint, formatter, dan unit test awal.

**Output:** frontend, backend, dan database dapat berjalan secara
konsisten.

### Fase 2 --- Login dan akses

-   [ ] Implementasikan login/session.
-   [ ] Implementasikan role dan authorization dependency.
-   [ ] Buat model pengguna.
-   [ ] Proteksi route frontend.
-   [ ] Terapkan ownership check pada dokumen dan percakapan.
-   [ ] Tambahkan audit log untuk event sensitif.
-   [ ] Uji akses lintas pengguna.

**Output:** pengguna dapat login dan data terisolasi berdasarkan izin.

### Fase 3 --- Dokumen dan OCR

-   [ ] Implementasikan upload tervalidasi.
-   [ ] Implementasikan storage service.
-   [ ] Buat model dokumen dan status ingest.
-   [ ] Implementasikan ekstraksi teks PDF.
-   [ ] Integrasikan PaddleOCR untuk dokumen scan.
-   [ ] Implementasikan cleaning dan chunking.
-   [ ] Tambahkan job status, retry, dan error handling.
-   [ ] Buat halaman daftar/status dokumen.

**Output:** dokumen dapat diunggah, diproses, dan dilacak.

### Fase 4 --- Embedding dan RAG dasar

-   [ ] Integrasikan embedding provider.
-   [ ] Simpan chunk dan embedding di PGVector.
-   [ ] Implementasikan similarity search dengan filter akses.
-   [ ] Buat context builder dan top-k retrieval.
-   [ ] Integrasikan LLM.
-   [ ] Implementasikan jawaban dengan sumber.
-   [ ] Buat dataset evaluasi awal dan baseline metrik.

**Output:** chatbot dapat menjawab pertanyaan berdasarkan dokumen yang
diizinkan.

### Fase 5 --- Agentic RAG dan n8n

-   [ ] Buat policy agent dan allowlist tool.
-   [ ] Batasi iterasi, timeout, token, dan tool calls.
-   [ ] Implementasikan fallback saat sumber tidak cukup.
-   [ ] Jalankan n8n dengan storage persisten dan encryption key stabil.
-   [ ] Buat workflow ingest dan workflow AI yang tervalidasi.
-   [ ] Amankan webhook dan akses editor.
-   [ ] Simpan export workflow tanpa credential di Git.
-   [ ] Uji workflow gagal, retry, dan idempotency.

**Output:** workflow AI dapat diorkestrasi dan diaudit dengan kontrol
akses.

### Fase 6 --- Fitur website

-   [ ] Buat UI chat dan status loading/error.
-   [ ] Buat daftar percakapan dan riwayat.
-   [ ] Tampilkan sitasi sumber.
-   [ ] Buat dashboard dokumen.
-   [ ] Buat panel admin minimum jika dibutuhkan.
-   [ ] Tambahkan pagination, validasi, dan aksesibilitas.
-   [ ] Jalankan end-to-end test.

**Output:** seluruh alur utama dapat dipakai melalui website.

### Fase 7 --- Security dan evaluasi

-   [ ] Terapkan batas request dan upload.
-   [ ] Review CORS, cookies, token/session, dan secret.
-   [ ] Implementasikan enkripsi file dan pengelolaan kunci.
-   [ ] Uji prompt injection, kebocoran data, dan akses lintas pengguna.
-   [ ] Evaluasi retrieval, faithfulness, sitasi, dan latency.
-   [ ] Jalankan dependency/security scan.
-   [ ] Dokumentasikan batasan yang belum terselesaikan.

**Output:** baseline keamanan dan kualitas terukur.

### Fase 8 --- Docker dan CI

-   [ ] Buat Dockerfile frontend dan backend.
-   [ ] Tambahkan health check, user non-root, dan resource constraints.
-   [ ] Buat Compose untuk stack yang diperlukan.
-   [ ] Tambahkan GitHub Actions CI.
-   [ ] Build dan scan image.
-   [ ] Setup registry dan tagging image.
-   [ ] Simpan secret di GitHub Actions Secrets/Environments.

**Output:** build dapat diulang dan divalidasi otomatis.

### Fase 9 --- VPS production

-   [ ] Siapkan VPS Ubuntu dan user admin.
-   [ ] Setup SSH key dan UFW.
-   [ ] Instal Docker Engine dan Compose.
-   [ ] Siapkan domain dan DNS.
-   [ ] Konfigurasi Nginx dan TLS.
-   [ ] Deploy container ke jaringan internal.
-   [ ] Jalankan migrasi dan smoke test.
-   [ ] Terapkan deployment pipeline dan rollback plan.
-   [ ] Pastikan database/n8n/monitoring tidak terbuka ke publik.

**Output:** website MVP tersedia melalui HTTPS.

### Fase 10 --- Operasional

-   [ ] Setup Prometheus/Grafana sesuai kebutuhan.
-   [ ] Setup Uptime Kuma.
-   [ ] Setup logging dan rotasi.
-   [ ] Setup Alertmanager dan notifikasi.
-   [ ] Setup backup database dan file ke lokasi off-site.
-   [ ] Uji restore dan dokumentasikan hasil.
-   [ ] Uji kapasitas dan sesuaikan resource limits.
-   [ ] Siapkan runbook incident dan prosedur pemulihan.

**Output:** MVP dapat dipantau, dipulihkan, dan dipelihara.

### Fase 11 --- Evaluasi dan future scaling

-   [ ] Tinjau metrik penggunaan, kualitas, latency, biaya, dan
    kapasitas.
-   [ ] Perbaiki retrieval/chunking berdasarkan hasil evaluasi.
-   [ ] Pisahkan worker/LLM ke host lain jika dibutuhkan.
-   [ ] Evaluasi OpenShift AI hanya jika kebutuhan MLOps enterprise
    sudah jelas.
-   [ ] Buat rencana migrasi dan integrasi tanpa menjadikan OpenShift AI
    dependency MVP.

**Output:** keputusan scaling berbasis bukti.

------------------------------------------------------------------------

## 30. Definition of Done

MVP dinyatakan siap untuk rilis awal jika seluruh hal berikut terpenuhi
atau pengecualiannya disetujui secara eksplisit:

### Fungsional

-   [ ] Pengguna dapat login.
-   [ ] Pengguna dapat mengunggah dokumen yang diizinkan.
-   [ ] Status pemrosesan dokumen terlihat.
-   [ ] Teks dokumen dapat diekstrak/OCR sesuai kebutuhan.
-   [ ] Embedding tersimpan di PGVector.
-   [ ] Chat dapat melakukan retrieval dengan filter hak akses.
-   [ ] Jawaban dapat menyertakan sumber yang benar.
-   [ ] Riwayat percakapan tersedia sesuai izin.
-   [ ] n8n menjalankan workflow yang telah diuji.

### Security

-   [ ] Secret tidak ada di Git atau frontend bundle.
-   [ ] Dokumen dan riwayat terisolasi berdasarkan hak akses.
-   [ ] HTTPS aktif.
-   [ ] UFW dan SSH key dikonfigurasi.
-   [ ] Port database dan service internal tidak terbuka ke publik.
-   [ ] Upload, rate limit, dan batas resource diterapkan.
-   [ ] Enkripsi file dan pengelolaan kunci terdokumentasi.
-   [ ] Audit event penting tersedia.

### Quality dan operasional

-   [ ] Test utama lulus.
-   [ ] Hasil evaluasi RAG direkam.
-   [ ] Health check dan log tersedia.
-   [ ] CI membangun image yang dapat direproduksi.
-   [ ] Deployment dan rollback terdokumentasi.
-   [ ] Backup off-site berjalan.
-   [ ] Restore pernah diuji.
-   [ ] Monitoring dan alert kritis aktif.
-   [ ] Batasan yang diketahui tercatat.

------------------------------------------------------------------------

## 31. Troubleshooting

### PGVector tidak ditemukan

-   Verifikasi image PostgreSQL yang dipakai menyertakan PGVector.
-   Periksa koneksi ke database yang benar.
-   Jalankan `CREATE EXTENSION IF NOT EXISTS vector;` dengan role yang
    memiliki izin.
-   Pastikan dimensi embedding cocok dengan tipe kolom vektor.
-   Periksa log container database.

### Container terus restart

``` bash
docker compose ps
docker compose logs --tail=200 <service>
docker stats
```

Periksa environment, health check, permission volume, koneksi
dependency, dan penggunaan RAM/CPU. Jangan langsung menghapus volume
database.

### Frontend tidak dapat menghubungi backend

-   Periksa base URL frontend dan backend.
-   Bedakan URL yang diakses browser dari URL internal antarkontainer.
-   Periksa CORS dan reverse proxy.
-   Periksa DNS, TLS, dan network Compose.
-   Jangan menggunakan `localhost` dari container untuk menunjuk
    container lain; gunakan nama service pada network yang sama.

### OCR atau ingest kehabisan memori

-   Kurangi concurrency.
-   Batasi ukuran dan jumlah halaman.
-   Pisahkan worker OCR dari API.
-   Monitor RAM dan durasi job.
-   Evaluasi model OCR dan strategi pemrosesan halaman.

### LLM timeout atau rate limit

-   Periksa provider/model, konektivitas, kuota, dan log teredaksi.
-   Terapkan timeout dan retry terbatas.
-   Batasi token dan concurrency.
-   Berikan fallback yang jelas di UI.
-   Hindari retry tanpa batas.

### n8n credential tidak terbaca setelah deploy

-   Pastikan `N8N_ENCRYPTION_KEY` sama dengan kunci saat credential
    dibuat.
-   Pastikan volume n8n persisten.
-   Jangan mengganti key sembarangan; lakukan prosedur migrasi
    credential yang direncanakan.

### Deployment gagal

-   Periksa status container, health check, log, dan image tag.
-   Pastikan environment production lengkap.
-   Periksa migrasi database dan kompatibilitas schema.
-   Rollback ke rilis sebelumnya sesuai prosedur.
-   Jangan menghapus volume data untuk menyelesaikan masalah deployment.

### Disk hampir penuh

-   Periksa penggunaan volume, log, image, database, upload, dan backup.
-   Terapkan rotasi log dan kebijakan retensi.
-   Hapus image tidak terpakai setelah memastikan aman.
-   Jangan menghapus data database atau backup secara manual tanpa
    kebijakan retensi.

------------------------------------------------------------------------

## 32. OpenShift AI --- Future Enterprise Scaling

OpenShift AI **bukan dependency wajib MVP**. Komponen ini masuk roadmap
apabila organisasi memerlukan platform MLOps enterprise, tata kelola
model, eksperimen, deployment model terkelola, atau scaling yang lebih
kompleks.

### 32.1 Kondisi untuk mengevaluasi OpenShift AI

-   Banyak model atau eksperimen perlu dikelola.
-   Training dan serving memerlukan resource khusus.
-   Diperlukan governance dan workflow MLOps lintas tim.
-   Kapasitas satu VPS tidak lagi cukup.
-   Infrastruktur Kubernetes/OpenShift dan keahlian operasional sudah
    tersedia atau layak diinvestasikan.

### 32.2 Prinsip transisi

-   Pertahankan kontrak API antar layanan.
-   Pisahkan adapter LLM dan embedding dari logika aplikasi.
-   Versioning model, prompt, pipeline, dan dataset.
-   Evaluasi kebutuhan migrasi database, storage, secret, dan
    observability secara terpisah.
-   Jangan menganggap seluruh komponen MVP harus dipindahkan ke
    OpenShift AI; tentukan komponen yang memang mendapat manfaat.

------------------------------------------------------------------------

## 33. Checklist Akhir

### Sebelum mulai coding

-   [ ] Scope website-only dikunci.
-   [ ] Pilihan LLM, embedding, storage, dan autentikasi diputuskan.
-   [ ] Repository dan environment lokal siap.
-   [ ] `.env.example` tersedia dan tidak berisi secret.

### Sebelum rilis

-   [ ] Alur upload → OCR/extraction → chunking → embedding → PGVector
    berfungsi.
-   [ ] Alur pertanyaan → retrieval → LLM → jawaban bersumber berfungsi.
-   [ ] n8n workflow aman dan dapat dipulihkan.
-   [ ] Login, RBAC, dan isolasi data diuji.
-   [ ] Docker image dan CI/CD berhasil.
-   [ ] VPS, Nginx, HTTPS, UFW, SSH key, dan resource limit
    dikonfigurasi.
-   [ ] Monitoring, logging, alerting, dan backup aktif.
-   [ ] Restore dan rollback diuji.
-   [ ] Dokumentasi operasional diperbarui.

### Sesudah rilis

-   [ ] Pantau kualitas jawaban, latency, biaya, dan kapasitas.
-   [ ] Review alert dan insiden.
-   [ ] Update dependency dan image secara terencana.
-   [ ] Uji restore secara berkala.
-   [ ] Evaluasi scaling berdasarkan data aktual.

------------------------------------------------------------------------

## 34. Referensi Dokumentasi

Gunakan dokumentasi resmi dan periksa kompatibilitas versi ketika
implementasi dilakukan.

-   Next.js: https://nextjs.org/docs
-   FastAPI: https://fastapi.tiangolo.com/
-   n8n: https://docs.n8n.io/
-   PostgreSQL: https://www.postgresql.org/docs/
-   PGVector: https://github.com/pgvector/pgvector
-   PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
-   Docker: https://docs.docker.com/
-   Docker Compose: https://docs.docker.com/compose/
-   GitHub Actions: https://docs.github.com/actions
-   GitHub Container Registry: https://docs.github.com/packages
-   Nginx: https://nginx.org/en/docs/
-   Let's Encrypt: https://letsencrypt.org/docs/
-   Ubuntu Server: https://ubuntu.com/server/docs
-   Prometheus: https://prometheus.io/docs/
-   Grafana: https://grafana.com/docs/
-   Loki: https://grafana.com/docs/loki/
-   Alertmanager:
    https://prometheus.io/docs/alerting/latest/alertmanager/
-   Uptime Kuma: https://github.com/louislam/uptime-kuma
-   OpenShift AI:
    https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed

------------------------------------------------------------------------

## Catatan Pemeliharaan Dokumen

README ini adalah **rencana implementasi dan baseline arsitektur**,
bukan klaim bahwa seluruh fitur sudah dibangun. Perbarui checklist dan
dokumentasi ketika komponen benar-benar selesai, catat perubahan
keputusan arsitektur, dan jangan menyatakan fitur production-ready
sebelum pengujian serta kontrol operasionalnya selesai.

------------------------------------------------------------------------

# README v2 --- Implementation-Ready Supplement

> Bagian ini melengkapi roadmap pada bagian sebelumnya dengan baseline
> implementasi yang lebih konkret. Semua nilai versi, nama model,
> kapasitas, dan threshold yang belum diverifikasi tetap harus diuji
> pada environment proyek sebelum dipromosikan ke production.

## 35. Baseline Versi dan Kebijakan Dependency

Versi final harus dikunci setelah smoke test di komputer developer dan
CI. Jangan mengklaim kombinasi versi sudah kompatibel hanya berdasarkan
rentang versi.

  -----------------------------------------------------------------------
  Komponen                Baseline yang           Kebijakan
                          disarankan              
  ----------------------- ----------------------- -----------------------
  Node.js                 LTS yang didukung versi Pin major/minor di
                          Next.js terpilih        `.nvmrc` atau
                                                  `.node-version`

  Next.js                 Versi stabil yang       Commit lockfile; jangan
                          dipilih saat            upgrade otomatis di
                          inisialisasi proyek     production

  Python                  Python 3.12 sebagai     Verifikasi dependency
                          kandidat baseline       OCR dan AI di
                                                  OS/container target

  FastAPI/Uvicorn         Versi stabil yang       Lock transitive
                          dikunci                 dependency

  PostgreSQL              PostgreSQL 17 sebagai   Gunakan image yang
                          baseline awal           cocok dengan PGVector

  PGVector                Versi yang tersedia di  Uji extension, tipe
                          image terpilih          vector, dan index

  Docker Compose          Plugin Compose v2       Catat versi di runbook

  n8n                     Versi image stabil yang Pertahankan encryption
                          dipilih dan dipin       key dan volume
                                                  persisten

  PaddleOCR               Versi yang lolos tes di OCR memiliki dependency
                          container target        sistem; uji CPU/RAM dan
                                                  model download

  Nginx                   Image stabil yang dipin Uji konfigurasi sebelum
                                                  reload

  GitHub Actions          Action versi            Pin action sesuai
                          major/commit yang       kebijakan keamanan
                          ditinjau                repository
  -----------------------------------------------------------------------

**Aturan rilis dependency:** setiap perubahan versi harus melewati
install bersih, lint, unit test, integration test, Docker build, dan
smoke test. Simpan lockfile frontend dan lockfile/pinned requirements
backend. Jangan menulis "fully compatible" sampai kombinasi yang
sebenarnya sudah diuji.

## 36. Struktur Implementasi yang Diperinci

``` text
ai-chatbot-mvp/
├── apps/web/
│   ├── app/
│   │   ├── login/
│   │   ├── chat/
│   │   ├── history/
│   │   ├── documents/
│   │   └── dashboard/
│   ├── components/
│   ├── lib/api.ts
│   ├── Dockerfile
│   └── package.json
├── services/api/
│   ├── app/
│   │   ├── api/routes/
│   │   ├── core/
│   │   ├── db/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── repositories/
│   │   ├── services/
│   │   └── workers/
│   ├── alembic/
│   ├── tests/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── n8n/workflows/
├── infra/nginx/
├── infra/monitoring/
├── infra/backup/
├── infra/scripts/
├── tests/evaluation/
├── .github/workflows/
├── .env.example
├── compose.yaml
└── README.md
```

## 37. Konfigurasi Environment yang Divalidasi

Simpan semua setting di satu objek konfigurasi backend, bukan membaca
environment secara acak di setiap modul. Contoh pola menggunakan
Pydantic Settings:

``` python
# services/api/app/core/config.py
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "development"
    app_secret_key: SecretStr
    database_url: SecretStr
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    llm_provider: str = "openai"
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    embedding_model: str
    embedding_dimension: int
    max_upload_size_mb: int = 25
    retrieval_top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("embedding_dimension")
    @classmethod
    def positive_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("embedding_dimension harus lebih dari nol")
        return value

    @field_validator("max_upload_size_mb", "retrieval_top_k")
    @classmethod
    def positive_limits(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Nilai batas harus lebih dari nol")
        return value

settings = Settings()
```

Pasang `pydantic-settings` jika menggunakan contoh ini. Konfigurasi
wajib seperti `app_secret_key`, `database_url`, `jwt_secret_key`,
`embedding_model`, dan `embedding_dimension` harus disediakan di
environment. Validasi tambahan harus memastikan provider LLM yang
dipilih memiliki kredensial atau endpoint yang sesuai. Jangan mencetak
nilai `SecretStr` ke log.

## 38. Docker Compose Development Terintegrasi

Berikut baseline Compose untuk service inti. Sesuaikan Dockerfile,
command, model AI, dan health check dengan implementasi yang benar-benar
tersedia. Jangan langsung mempublikasikan port database atau n8n ke
internet.

``` yaml
name: ai-chatbot-mvp

services:
  postgres:
    image: pgvector/pgvector:pg17
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 10
    networks: [backend]

  api:
    build:
      context: ./services/api
    restart: unless-stopped
    env_file: .env
    environment:
      DATABASE_URL: ${DATABASE_URL}
      APP_ENV: ${APP_ENV:-development}
    depends_on:
      postgres:
        condition: service_healthy
    expose: ["8000"]
    networks: [backend, frontend]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
      interval: 20s
      timeout: 5s
      retries: 5

  web:
    build:
      context: ./apps/web
      args:
        NEXT_PUBLIC_API_BASE_URL: ${NEXT_PUBLIC_API_BASE_URL:-/api}
    restart: unless-stopped
    depends_on:
      api:
        condition: service_healthy
    expose: ["3000"]
    networks: [frontend]

  n8n:
    image: n8nio/n8n:${N8N_VERSION}
    restart: unless-stopped
    environment:
      N8N_HOST: ${N8N_HOST:-localhost}
      N8N_PORT: 5678
      N8N_PROTOCOL: ${N8N_PROTOCOL:-http}
      N8N_ENCRYPTION_KEY: ${N8N_ENCRYPTION_KEY}
      N8N_DIAGNOSTICS_ENABLED: "false"
      N8N_PERSONALIZATION_ENABLED: "false"
      GENERIC_TIMEZONE: ${GENERIC_TIMEZONE:-Asia/Jakarta}
    volumes:
      - n8n_data:/home/node/.n8n
    expose: ["5678"]
    networks: [backend]
    profiles: ["automation"]

  minio:
    image: minio/minio:${MINIO_VERSION}
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - minio_data:/data
    expose: ["9000", "9001"]
    networks: [backend]
    profiles: ["object-storage"]

networks:
  frontend:
  backend:
    internal: true

volumes:
  postgres_data:
  n8n_data:
  minio_data:
```

**Catatan penting:** service `web` membutuhkan Dockerfile yang sesuai;
`NEXT_PUBLIC_API_BASE_URL` merupakan nilai build-time dan tidak boleh
berisi secret. Jika menggunakan Next.js standalone output, Dockerfile
harus menyalin artefak standalone dan static assets yang benar. Profil
`automation` dan `object-storage` hanya aktif jika dipanggil, misalnya
`docker compose --profile automation up -d`. Untuk akses development
dari host, tambahkan port binding localhost secara sengaja; jangan
menambahkan port publik pada production.

### 38.1 Urutan startup development

``` bash
cp .env.example .env
docker compose config
docker compose up -d postgres
docker compose ps
# Jalankan migration dari container API atau environment Python proyek.
docker compose up -d api web
docker compose --profile automation up -d n8n
docker compose ps
docker compose logs -f api web
```

Sebelum perintah dijalankan, lengkapi `.env` dan `N8N_VERSION`; jika
memakai object storage, lengkapi juga `MINIO_VERSION` serta
credential-nya. Startup pertama kali tidak dianggap sukses sampai
migration, health check, dan smoke test lulus.

Shutdown:

``` bash
docker compose stop
```

Untuk menghapus container dan network tanpa menghapus data persisten:

``` bash
docker compose down
```

Jangan gunakan `docker compose down -v` kecuali memang ingin menghapus
volume data dan sudah memastikan tidak ada data yang perlu
dipertahankan.

## 39. Dockerfile Frontend dan Backend

### 39.1 Backend

``` dockerfile
# services/api/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Jika OCR membutuhkan system package atau model tambahan, tambahkan
secara eksplisit dan pin versi yang sesuai. Gunakan dependency lock yang
sudah diuji.

### 39.2 Frontend

Atur `output: "standalone"` pada konfigurasi Next.js jika memilih
deployment standalone. Contoh berikut mengasumsikan output tersebut
sudah diaktifkan:

``` dockerfile
# apps/web/Dockerfile
FROM node:22-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_BASE_URL=/api
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
USER node
EXPOSE 3000
CMD ["node", "server.js"]
```

Jika folder `public` belum dibuat, buat folder tersebut atau ubah
langkah `COPY` sesuai struktur proyek. Pin image Node yang sudah diuji;
jangan menganggap tag ini final tanpa build dan test pada CI. Tambahkan
`.dockerignore` pada kedua aplikasi.

## 40. Schema Data, Migration, dan Kontrak API

### 40.1 Entitas minimum

Implementasikan dan versioning entitas berikut:

-   `users` dan role/permission.
-   `documents` beserta owner/scope akses, checksum, status, lokasi
    storage, dan timestamp.
-   `document_chunks` beserta nomor halaman dan metadata.
-   `document_embeddings` atau kolom vector di tabel chunk, termasuk
    model dan versi embedding.
-   `conversations`, `messages`, dan `message_sources`.
-   `ingestion_jobs` dengan status, retry count, dan error teredaksi.
-   `audit_logs` untuk event sensitif.

Gunakan foreign key, index, constraint, dan aturan delete yang
eksplisit. Tetapkan apakah penghapusan dokumen bersifat cascading, soft
delete, atau penghapusan bertahap sesuai kebijakan retensi. Jangan
menyimpan embedding tanpa metadata yang memungkinkan validasi akses
dokumen.

### 40.2 Migration

``` bash
alembic revision --autogenerate -m "create core tables"
alembic upgrade head
alembic current
```

Periksa file migration sebelum menjalankannya. Jalankan migration pada
database development kosong dalam CI. Pada production, lakukan backup
sebelum migration berisiko dan gunakan strategi yang kompatibel dengan
rollback aplikasi.

### 40.3 Kontrak endpoint minimum

  -------------------------------------------------------------------------------------
  Endpoint                              Keamanan                Respons penting
  ------------------------------------- ----------------------- -----------------------
  `POST /api/v1/auth/login`             Rate limited            Session/token, tanpa
                                                                mengembalikan secret

  `GET /api/v1/auth/me`                 Login                   Identitas dan role

  `POST /api/v1/documents`              Login + kuota           ID dokumen dan status
                                                                `queued`

  `GET /api/v1/documents`               Login + filter akses    Dokumen milik/diizinkan

  `GET /api/v1/documents/{id}`          Ownership/RBAC          Metadata dan status

  `DELETE /api/v1/documents/{id}`       Ownership/RBAC          Status penghapusan

  `POST /api/v1/chat`                   Login + rate limit      Jawaban, sumber,
                                                                conversation ID

  `GET /api/v1/conversations`           Login + filter owner    Daftar percakapan

  `GET /api/v1/conversations/{id}`      Owner/RBAC              Pesan dan sumber

  `DELETE /api/v1/conversations/{id}`   Owner/RBAC              Status penghapusan

  `GET /health`                         Informasi minimal       Liveness

  `GET /ready`                          Internal/terbatas jika  Readiness dependency
                                        perlu                   
  -------------------------------------------------------------------------------------

Gunakan schema Pydantic terpisah untuk request dan response. Kembalikan
status HTTP yang sesuai: `401` untuk belum terautentikasi, `403` untuk
tidak berizin, `413` untuk payload terlalu besar, `422` untuk input
tidak valid, dan `429` untuk rate limit. Jangan mengungkap apakah
dokumen milik pengguna lain ada atau tidak.

## 41. Pipeline Job Ingestion

Gunakan state machine yang eksplisit:

``` text
uploaded -> queued -> processing -> completed
                              \-> failed
                              \-> needs_review
```

Setiap job setidaknya menyimpan `job_id`, `document_id`, `status`,
`attempt_count`, `created_at`, `started_at`, `finished_at`, dan error
code yang tidak sensitif.

Langkah worker:

1.  Claim job secara aman agar tidak diproses dua worker sekaligus.
2.  Periksa status dan idempotency.
3.  Validasi file serta checksum.
4.  Ekstrak teks; jalankan OCR hanya jika diperlukan.
5.  Bersihkan teks dan pertahankan nomor halaman.
6.  Chunk teks dan buat embedding.
7.  Simpan chunk/vector dalam transaksi yang konsisten.
8.  Tandai job selesai.
9.  Jika gagal, simpan kategori error dan lakukan retry terbatas dengan
    backoff.
10. Jika jumlah retry habis, tandai gagal dan sediakan tindakan
    reprocess yang diotorisasi.

Untuk MVP, worker dapat berupa proses terpisah. Jika memakai
Redis/queue, tambahkan sebagai service yang terdokumentasi dan berikan
persistence/monitoring sesuai kebutuhan. Jangan menambah dependency
queue tanpa keputusan arsitektur dan pengujian.

## 42. Kontrak Agentic RAG dan Tool Registry

Buat interface yang dapat diuji, terpisah dari provider model:

``` python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class RetrievedSource:
    document_id: str
    chunk_id: str
    text: str
    page: int | None
    score: float

@dataclass
class RAGAnswer:
    answer: str
    sources: list[RetrievedSource]
    status: str

class Retriever(Protocol):
    async def search(
        self, query: str, *, user_id: str, top_k: int
    ) -> list[RetrievedSource]: ...

class LLMClient(Protocol):
    async def generate(self, *, system_prompt: str, question: str,
                       context: str) -> str: ...
```

Implementasi produksi harus menambahkan timeout, penanganan provider
error, batas token, tracing, dan filter akses. Contoh ini adalah kontrak
interface, bukan RAG service lengkap.

### 42.1 Agent policy

-   Tool yang terdaftar memiliki nama, deskripsi, schema input,
    otorisasi, timeout, dan batas pemanggilan.
-   Tool database tidak menerima SQL bebas dari model.
-   Tool storage tidak menerima path file sembarang.
-   Tool eksternal memakai allowlist host dan mencegah SSRF.
-   Batasi iterasi agent dan total waktu per pertanyaan.
-   Dokumen hasil retrieval diperlakukan sebagai data tidak tepercaya.
-   Saat sumber tidak cukup, jawab bahwa informasi belum ditemukan.
-   Simpan sumber yang benar-benar mendukung jawaban, bukan semua chunk
    yang kebetulan diambil.

## 43. Workflow n8n yang Dapat Diimpor

Simpan workflow export JSON di `n8n/workflows/`. Workflow harus dibuat
pada versi n8n yang dipin, diekspor dari editor, diperiksa, lalu diuji
pada environment development. Jangan mengarang ID credential atau
mengekspor secret.

### 43.1 Kontrak workflow yang disepakati

**Document ingest:** trigger terautentikasi → validasi `document_id` →
panggil endpoint internal backend untuk memulai/memeriksa job → cabang
sukses/gagal → notifikasi opsional.

**AI workflow:** trigger internal yang diautentikasi → validasi input →
panggil endpoint Agentic RAG → validasi response → kembalikan hasil ke
pemanggil.

**Operational alert:** trigger dari monitoring → deduplikasi →
notifikasi tanpa secret atau konten dokumen.

### 43.2 Prosedur membuat workflow

1.  Jalankan n8n pada profile development.
2.  Buat credential melalui UI menggunakan secret development.
3.  Buat node trigger yang sesuai, node HTTP Request ke backend, IF
    node, dan node notifikasi jika diperlukan.
4.  Terapkan authentication dan validasi payload pada trigger.
5.  Tetapkan timeout, retry terbatas, dan error branch.
6.  Uji input valid, input invalid, backend timeout, dan retry.
7.  Export workflow JSON tanpa credential.
8.  Simpan export di Git dan catat versi n8n yang digunakan.
9.  Import ke staging/production, lalu pasang credential melalui
    mekanisme environment target.
10. Batasi akses editor dan webhook.

**Catatan:** workflow export harus berasal dari workflow yang
benar-benar dibuat dan diuji pada versi n8n target. Jangan menganggap
contoh JSON generik dapat langsung diimpor tanpa verifikasi node
type/version, URL, dan credential.

## 44. Autentikasi, RBAC, dan Enkripsi File --- Acceptance Criteria

### Authentication/session

-   Password, jika dikelola sendiri, di-hash dengan algoritma password
    hashing modern; jangan menyimpan password plaintext.
-   Session/token memiliki expiry, revocation/logout yang terdefinisi,
    dan secret kuat.
-   Cookie browser menggunakan `HttpOnly`, `Secure`, dan `SameSite` yang
    sesuai.
-   Login memiliki rate limit dan respons yang tidak memudahkan
    enumerasi akun.
-   Secret tidak ditulis ke log atau dikirim ke frontend.

### RBAC dan kepemilikan data

-   Semua query dokumen dan percakapan menerapkan owner/scope filter di
    backend.
-   Akses ke sumber retrieval divalidasi sebelum teks diberikan ke LLM.
-   Endpoint admin memerlukan role yang sesuai.
-   Test wajib memastikan User A tidak bisa membaca, menghapus, atau
    memperoleh hasil retrieval dari data User B.

### Enkripsi file

-   Gunakan authenticated encryption dari library yang terpelihara,
    misalnya AES-256-GCM bila sesuai.
-   Kunci dibuat dengan cryptographically secure random generator dan
    disimpan di secret manager atau mekanisme perlindungan yang sesuai.
-   Nonce unik sesuai persyaratan algoritma; simpan versi format
    ciphertext agar dapat dimigrasikan.
-   Uji dekripsi, ciphertext yang dimodifikasi, key salah, dan rotasi
    key.
-   Kunci dekripsi tidak disimpan bersama ciphertext tanpa perlindungan.
-   Rencana pemulihan harus mencakup kunci yang diperlukan; backup
    ciphertext tanpa kunci yang dapat dipulihkan tidak berguna.

### Prompt injection dan tool abuse

-   Uji dokumen yang berisi instruksi untuk mengabaikan system prompt.
-   Uji permintaan untuk mengambil dokumen milik pengguna lain.
-   Uji permintaan untuk membuka environment variable, file server, dan
    credential.
-   Uji tool dengan parameter invalid dan URL internal.
-   Agent harus menolak atau mengabaikan instruksi yang tidak berwenang,
    dan tidak boleh memperluas hak aksesnya.

## 45. GitHub Actions CI/CD --- Persyaratan Workflow

Simpan dua workflow terpisah: `.github/workflows/ci.yml` dan
`.github/workflows/deploy.yml`.

### CI wajib

1.  Trigger pada pull request dan push.
2.  Install frontend dari lockfile menggunakan `npm ci`.
3.  Jalankan lint, type check, test, dan build frontend.
4.  Setup Python sesuai versi proyek.
5.  Install backend dari dependency yang dikunci.
6.  Jalankan lint/format check dan unit test.
7.  Jalankan test integrasi menggunakan dependency service yang
    diperlukan.
8.  Jalankan dependency dan container image scan.
9.  Build image untuk setiap service yang dirilis.
10. Gagal jika ada test atau pemeriksaan wajib yang gagal.

### CD wajib

1.  Hanya berjalan dari branch/tag rilis yang disetujui.
2.  Gunakan environment production dengan approval bila diperlukan.
3.  Build image dengan tag commit SHA atau digest immutable.
4.  Scan image sebelum publish.
5.  Push ke GHCR/registry.
6.  Login ke VPS dengan deployment key khusus atau metode deployment
    yang disetujui.
7.  Verifikasi SSH host key.
8.  Backup sebelum migrasi yang berisiko.
9.  Deploy image dan konfigurasi yang telah divalidasi.
10. Tunggu health check dan jalankan smoke test.
11. Hentikan promosi rilis jika health check gagal; jalankan rollback
    sesuai runbook.
12. Catat versi image dan hasil deployment.

Pin versi GitHub Actions sesuai kebijakan keamanan. Gunakan
`permissions` minimum, jangan memberikan `write-all`, dan jangan
memasukkan secret pada log. File workflow final harus disesuaikan dengan
nama image, branch, direktori deploy, serta metode akses VPS yang
sebenarnya.

## 46. Nginx dan TLS --- Checklist Konfigurasi

Konfigurasi production harus memenuhi hal berikut:

-   Domain dan DNS mengarah ke VPS yang benar.
-   Hanya Nginx yang menerima trafik HTTP/HTTPS publik.
-   Nginx meneruskan request ke service melalui jaringan
    Docker/internal.
-   Batas ukuran upload selaras dengan batas backend.
-   Timeout cukup untuk request chat dan tidak membiarkan koneksi
    menggantung tanpa batas.
-   Jika memakai streaming, header dan buffering dikonfigurasi sesuai
    protokol.
-   Header forwarding ditetapkan dengan benar.
-   HTTP diarahkan ke HTTPS setelah sertifikat valid.
-   Sertifikat diperbarui otomatis dan renewal diuji.
-   Editor n8n dan dashboard monitoring tidak terbuka untuk umum.
-   Konfigurasi diuji sebelum reload dengan `nginx -t`.

Jangan menyalin konfigurasi generik ke production tanpa mengganti
domain, port, path, dan aturan akses yang sesuai.

## 47. Deployment Script dan Rollback

Script deployment harus berhenti saat terjadi error, memvalidasi
konfigurasi, dan tidak menghapus volume. Alur operasional yang
disarankan:

``` text
Preflight
  -> Verify environment and image tags
  -> Backup database/data if needed
  -> Pull immutable image tags
  -> Validate Compose configuration
  -> Run compatible database migration
  -> Apply deployment
  -> Wait for health checks
  -> Run smoke tests
  -> Mark release successful
  -> If failed: restore previous app image/config and investigate
```

### Rollback requirements

-   Simpan image/tag rilis sebelumnya.
-   Simpan salinan konfigurasi deployment non-secret yang terkait rilis.
-   Gunakan migration yang backward-compatible jika memungkinkan.
-   Jangan otomatis menurunkan schema database tanpa rencana migration
    downgrade yang diuji.
-   Bedakan rollback aplikasi dari pemulihan data.
-   Dokumentasikan siapa yang boleh menjalankan rollback dan bagaimana
    hasilnya diverifikasi.

## 48. Monitoring, Alert Rules, Log Rotation, Backup, dan Restore

### Monitoring

Dashboard harus mencakup host CPU/RAM/disk, container restart, uptime
endpoint, latency/error API, koneksi database, kegagalan ingest,
latency/error LLM, dan status backup.

### Alert rules

Buat aturan yang terukur untuk: - Endpoint tidak tersedia. - Error rate
atau latency melewati baseline yang disepakati. - Disk hampir penuh. -
Memori tinggi/OOM atau container restart berulang. - Database tidak
siap. - Ingest gagal berulang. - Provider LLM timeout/rate limit
meningkat. - Backup gagal atau umur backup melewati batas. - Sertifikat
TLS mendekati kedaluwarsa.

Ambang harus dikalibrasi dari baseline dan diuji dengan simulasi alert.

### Log rotation

-   Atur batas ukuran dan jumlah file log Docker.
-   Atur retensi log terpusat.
-   Redaksi token, cookie, API key, prompt sensitif, dan data pribadi.
-   Pastikan disk tetap cukup ketika log system sedang bermasalah.
-   Jangan mengaktifkan debug logging permanen di production.

### Backup dan restore

Backup meliputi database, file yang harus dipertahankan, konfigurasi
non-secret, export workflow n8n, dan material pemulihan secret/kunci
yang disimpan secara terpisah dan aman.

Uji restore minimal: 1. Buat backup dengan timestamp dan catat hasilnya.
2. Pulihkan ke environment terisolasi. 3. Verifikasi schema dan jumlah
data yang masuk akal. 4. Verifikasi file dapat dibaca/dekripsi dengan
kunci yang benar. 5. Jalankan query retrieval dan smoke test aplikasi.
6. Catat durasi restore dan kekurangan prosedur. 7. Jangan menganggap
backup valid sampai restore berhasil.

## 49. Test Suite dan Evaluasi RAG

### Unit tests

-   Validasi request dan file.
-   Chunking dan metadata halaman.
-   Filter ownership.
-   Pembentukan konteks dan batas token.
-   Policy tool agent.
-   Penanganan timeout dan provider error.
-   Enkripsi/dekripsi dan penolakan ciphertext yang dimodifikasi.

### Integration tests

-   API ↔ PostgreSQL/PGVector.
-   Upload → job → extraction/OCR → chunk → embedding.
-   Retrieval dengan filter akses.
-   Backend ↔ LLM adapter.
-   Backend ↔ n8n webhook.
-   Storage ↔ database metadata.

### End-to-end tests

1.  Login.
2.  Upload dokumen.
3.  Tunggu ingest selesai.
4.  Ajukan pertanyaan yang jawabannya ada di dokumen.
5.  Verifikasi jawaban dan sumber.
6.  Ajukan pertanyaan yang tidak ada di dokumen.
7.  Verifikasi sistem menyatakan keterbatasan.
8.  Pastikan pengguna lain tidak dapat mengakses dokumen/percakapan
    tersebut.

### Dataset evaluasi

Sediakan file JSONL/CSV non-sensitif berisi `question`,
`expected_answer`, `expected_source_ids`, `user_scope`, dan kategori
pertanyaan. Pisahkan data development dari data evaluasi akhir untuk
menghindari tuning terhadap test set.

Metrik yang dicatat: - Recall@k atau hit rate sumber relevan. - Citation
precision/correctness. - Faithfulness terhadap konteks. - Answer
completeness. - Latency median dan p95. - Error/timeout rate. - Fallback
correctness ketika tidak ada sumber relevan.

Tetapkan target setelah baseline; jangan mengarang angka keberhasilan
sebelum pengukuran.

### Load test

Uji skenario chat bersamaan dan ingest dokumen dengan ukuran yang
representatif. Pantau p95 latency, error rate, CPU, RAM, disk, database
connections, dan antrean job. Mulai dari beban kecil, naikkan bertahap,
dan hentikan pengujian jika resource melewati batas aman.

## 50. Runbook Insiden dan Pemulihan

### API tidak tersedia

1.  Periksa health check dan status container.
2.  Periksa log terbaru tanpa menyalin secret.
3.  Periksa database, jaringan, dan provider eksternal.
4.  Tentukan apakah perlu rollback.
5.  Catat waktu mulai, dampak, dan tindakan.

### Database penuh/tidak siap

1.  Periksa disk dan koneksi.
2.  Jangan menghapus volume.
3.  Pastikan backup terbaru valid.
4.  Identifikasi query, log, atau data yang menyebabkan masalah.
5.  Pulihkan layanan dengan prosedur yang disetujui dan catat tindakan.

### Secret atau credential terpapar

1.  Cabut/rotasi credential terdampak.
2.  Batasi akses dan periksa audit log.
3.  Perbarui secret store dan layanan terkait.
4.  Validasi integrasi setelah rotasi.
5.  Dokumentasikan insiden tanpa menaruh secret pada laporan.

### Backup gagal

1.  Periksa ruang penyimpanan, kredensial, dan status job.
2.  Jangan menghapus backup terakhir yang valid.
3.  Jalankan ulang secara terkendali.
4.  Verifikasi backup baru dan jadwalkan restore test.

### Patch dan update

-   Review release notes dan advisori keamanan.
-   Update branch terpisah.
-   Jalankan CI dan staging smoke test.
-   Backup sebelum perubahan berisiko.
-   Deploy bertahap dan siapkan rollback.
-   Catat versi lama/baru dan hasil verifikasi.

## 51. Load Test dan Kriteria Kelulusan MVP

MVP baru dapat dinyatakan lulus setelah tim menyepakati target
kuantitatif. Ukur dan dokumentasikan:

-   p50/p95 latency untuk request chat.
-   Throughput dan jumlah pengguna bersamaan yang diuji.
-   Tingkat error dan timeout.
-   Waktu ingest untuk dokumen dengan ukuran/halaman yang ditetapkan.
-   Recall@k dan kebenaran sumber.
-   Perilaku saat dokumen tidak mengandung jawaban.
-   CPU, RAM, disk, dan database connections saat beban.
-   Waktu pemulihan dari backup dan target RPO/RTO.

Jangan menetapkan klaim SLA atau kapasitas produksi hanya dari
spesifikasi nominal VPS; hasilnya harus didukung benchmark pada stack
aktual.

## 52. Kriteria Evaluasi OpenShift AI

OpenShift AI tetap merupakan **opsi future scaling**, bukan service
wajib pada MVP. Mulai evaluasi bila ada kebutuhan terukur untuk model
lifecycle, training/fine-tuning, experiment tracking, model serving
terkelola, governance, atau skala multi-team.

Sebelum migrasi, dokumentasikan: - Problem yang tidak lagi dapat
diselesaikan secara efisien oleh MVP. - Komponen yang benar-benar perlu
dipindahkan. - Kebutuhan cluster, GPU/CPU, storage, network, dan
observability. - Biaya operasional dan kompetensi tim. - Strategi
migrasi model, endpoint, secret, data, dan rollback. - Uji
kompatibilitas dan kriteria keberhasilan transisi.

Jangan memindahkan seluruh aplikasi ke platform enterprise hanya karena
platform tersedia; keputusan harus didukung kebutuhan dan evaluasi
biaya/manfaat.

## 53. Checklist Gate Rilis

-   [ ] Konfigurasi environment divalidasi tanpa mencetak secret.
-   [ ] Docker Compose valid dan semua service kritis sehat.
-   [ ] Migration dapat diterapkan pada database kosong.
-   [ ] Login dan RBAC diuji.
-   [ ] Dokumen dan percakapan terisolasi antar pengguna.
-   [ ] Ingest, OCR, chunking, embedding, dan retrieval berhasil.
-   [ ] Jawaban RAG menampilkan sumber yang benar.
-   [ ] Agent/tool dibatasi dan prompt injection diuji.
-   [ ] Workflow n8n diimpor serta diuji pada versi target.
-   [ ] Unit, integration, security, dan E2E test lulus.
-   [ ] Image dibangun, dipindai, dan diberi tag immutable.
-   [ ] CI/CD menggunakan permission minimum dan SSH host verification.
-   [ ] Domain, Nginx, HTTPS, UFW, dan SSH key diverifikasi.
-   [ ] Port internal tidak terbuka ke publik.
-   [ ] Resource limit dan log rotation aktif.
-   [ ] Monitoring, alerting, backup, dan restore telah diuji.
-   [ ] Deployment dan rollback diuji di staging.
-   [ ] Batasan dan hasil evaluasi didokumentasikan.
-   [ ] OpenShift AI tetap diperlakukan sebagai roadmap, bukan
    dependency MVP.

------------------------------------------------------------------------

## Catatan status implementasi

Penambahan bagian ini melengkapi instruksi dan contoh baseline, tetapi
tidak berarti seluruh file konfigurasi atau workflow telah dibuat,
dijalankan, dan lolos uji pada environment pengguna. Sebelum production,
tim harus membuat artefak final yang sesuai repository aktual, mengunci
versi yang telah diuji, mengimpor dan menguji workflow n8n pada versi
target, serta menjalankan test dan restore rehearsal.
