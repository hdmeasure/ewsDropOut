# Diagram Alir

## 1. Penyiapan data → model final

```mermaid
flowchart TD
    A[Data administratif siswa<br/>Dapodik] --> P[Penyiapan dataset]
    B[Skor asesmen standar<br/>ASPD numerasi] --> P
    C[Survei mutu sekolah<br/>Sulingjar] --> P
    L[Label: DO vs Non-DO] --> P

    P --> M0[m0: gabung + encoding<br/>buang variabel artefak]
    M0 --> SPLIT[Split per sekolah<br/>GroupKFold by NPSN]
    SPLIT --> M1[m1: Nested CV<br/>feature selection voting + tuning]
    M1 --> M2[m2: refit + kalibrasi<br/>+ koreksi prior + threshold]
    M2 --> M3[m3: evaluasi<br/>hold-out test]
    M2 --> M4[m4: interpretasi SHAP]
    M2 --> FINAL[(Model final<br/>aspd_num + tanpa_aspd)]
```

## 2. Pemilihan fitur (hybrid voting, di dalam fold CV)

```mermaid
flowchart LR
    X[Kandidat fitur] --> V1[Signifikansi<br/>regresi logistik p&lt;0.05]
    X --> V2[Random Forest<br/>importance &gt; median]
    X --> V3[SHAP XGBoost<br/>&#124;value&#124; &gt; median]
    V1 --> VOTE{Terpilih bila<br/>muncul di &ge;2 dari 3}
    V2 --> VOTE
    V3 --> VOTE
    VOTE --> SEL[Fitur terpilih]
```

## 3. Penanganan ketidakseimbangan & proporsi rasional

```mermaid
flowchart TD
    T[Data latih enriched<br/>prevalensi DO tinggi] --> CW[class_weight / scale_pos_weight<br/>TANPA SMOTE]
    CW --> CAL[Kalibrasi probabilitas<br/>sigmoid / isotonic]
    CAL --> PRIOR[Koreksi prior<br/>pi_train -&gt; pi_pop]
    PRIOR --> THR[Threshold = base rate<br/>proporsi ter-flag = pi_pop]
    THR --> OUT[Daftar siswa berisiko<br/>proporsi rasional]
```

## 4. Implementasi tiered + ekspor platform

```mermaid
flowchart TD
    NEW[Siswa baru / populasi penuh] --> CHK{Skor numerasi<br/>tersedia?}
    CHK -- Ya --> P1[Model utama: aspd_num]
    CHK -- Tidak --> CHK2{Variabel inti<br/>lengkap?}
    CHK2 -- Ya --> P2[Model fallback: tanpa_aspd]
    CHK2 -- Tidak --> ND[Data Tidak Lengkap<br/>tidak diprediksi]
    P1 --> R[prob_do + risiko_do + alasan]
    P2 --> R
    R --> EXP[Ekspor: .rds / .json / .joblib]
    EXP --> PLAT[(Platform)]
```

> Catatan: Sulingjar adalah variabel level sekolah, sehingga split & cross-validation
> dilakukan **per sekolah** (GroupKFold) untuk mencegah kebocoran data.
