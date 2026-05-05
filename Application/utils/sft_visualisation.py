import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Helper – identyczna logika permutacji co w sft.py
# ---------------------------------------------------------------------------
def _build_permutation(H, W, rng_seed=42):
    """
    Tworzy permutację wierszy i kolumn, aby rozproszyć sygnał przestrzennie.
    Dydaktycznie: Zwraca również parametry 'a' i 'b' do późniejszej wizualizacji.
    """
    rng = np.random.default_rng(rng_seed)
    a = int(rng.integers(1, max(H // 2, 2)) * 2 + 1)
    b = int(rng.integers(0, H))
    row_perm = (a * np.arange(H) + b) % H
    col_perm = (a * np.arange(W) + b) % W
    inv_row  = np.argsort(row_perm)
    inv_col  = np.argsort(col_perm)
    return row_perm, col_perm, inv_row, inv_col, a, b


def _log_spectrum(spectrum):
    """Bezpieczny log-amplitudowy widok widma (przesunięty do centrum)."""
    shifted = np.fft.fftshift(np.abs(spectrum))
    return np.log(1e-5 + shifted)


# ---------------------------------------------------------------------------
# KROK 1 – Permutacja wejściowego obrazu
# ---------------------------------------------------------------------------
def show_sft_step1_permutation(cropped_image):
    """
    Krok 1: losowa permutacja wierszy i kolumn.
    Cel: randomizacja sygnału, żeby energia rozłożyła się równomiernie
    po kubełkach FFT i uniknąć aliasingu w fazie subsamplingu.
    """
    H, W = cropped_image.shape
    # Poprawka: Odbieramy parametry 'a' i 'b' bez ponownego wywoływania funkcji
    row_perm, col_perm, _, _, a, b = _build_permutation(H, W)
    permuted = cropped_image[np.ix_(row_perm, col_perm)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SFT – Krok 1: Losowa permutacja wierszy i kolumn",
                 fontsize=13, fontweight='bold')

    axes[0].imshow(cropped_image, cmap='gray')
    axes[0].set_title("Oryginał")
    axes[0].axis('off')

    # Poprawka: Czysta i skalowalna wizualizacja wektora permutacji
    axes[1].imshow(row_perm.reshape(-1, 1), cmap='plasma', aspect='auto')
    axes[1].set_title(f"Mapa permutacji wierszy\n(a={a}, b={b})")
    axes[1].set_xlabel("(Rozkład źródłowych indeksów)")
    axes[1].set_ylabel("Nowy indeks wiersza")
    axes[1].set_xticks([]) # Ukrywamy oś X dla czytelności

    axes[2].imshow(permuted, cmap='gray')
    axes[2].set_title("Obraz po permutacji")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# KROK 2 – Pełne FFT2 spermutowanego sygnału (widmo resztkowe)
# ---------------------------------------------------------------------------
def show_sft_step2_full_spectrum(cropped_image):
    """
    Krok 2: FFT2 spermutowanego obrazu = pełne widmo resztkowe,
    od którego SFT będzie kolejno odejmować znalezione składowe.
    """
    H, W = cropped_image.shape
    row_perm, col_perm, *_ = _build_permutation(H, W)
    permuted = cropped_image[np.ix_(row_perm, col_perm)]
    residual = np.fft.fft2(permuted)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("SFT – Krok 2: Pełne FFT2 spermutowanego obrazu (widmo resztkowe)",
                 fontsize=13, fontweight='bold')

    axes[0].imshow(permuted, cmap='gray')
    axes[0].set_title("Wejście: spermutowany obraz")
    axes[0].axis('off')

    im1 = axes[1].imshow(_log_spectrum(residual), cmap='inferno')
    axes[1].set_title("Widmo amplitudy – log|F(u,v)|\n(centrum = DC, brzegi = wysokie częst.)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(np.fft.fftshift(np.angle(residual)), cmap='hsv')
    axes[2].set_title("Widmo fazy – arg(F(u,v))")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# KROK 3 – Haszowanie do kubełków (subsampling + lokalne FFT2)
# ---------------------------------------------------------------------------
def show_sft_step3_bucket_hashing(cropped_image, keep_fraction=0.05):
    """
    Krok 3: subsampling sygnału co stride_h / stride_w pikseli
    i lokalne FFT2 próbki → kubełki energii.
    Każdy kubełek agreguje grupę częstotliwości.
    """
    H, W = cropped_image.shape
    k        = max(int(H * W * keep_fraction), 1)
    n_buckets = int(2 ** np.ceil(np.log2(max(4 * k, 64))))
    stride_h = max(H // n_buckets, 1)
    stride_w = max(W // n_buckets, 1)

    row_perm, col_perm, *_ = _build_permutation(H, W)
    permuted  = cropped_image[np.ix_(row_perm, col_perm)]
    residual  = np.fft.fft2(permuted)

    signal_2d   = np.fft.ifft2(residual).real   # = permuted
    subsampled  = signal_2d[::stride_h, ::stride_w]
    bucket_spec = np.fft.fft2(subsampled)
    power_map   = np.abs(bucket_spec) ** 2

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"SFT – Krok 3: Haszowanie do kubełków  "
        f"(n_buckets={n_buckets}, stride_h={stride_h}, stride_w={stride_w})",
        fontsize=12, fontweight='bold')

    axes[0].imshow(permuted, cmap='gray')
    axes[0].set_title("Spermutowany obraz")
    axes[0].axis('off')

    # Zaznacz siatkę próbkowania
    axes[1].imshow(permuted, cmap='gray', alpha=0.5)
    ys = np.arange(0, H, stride_h)
    xs = np.arange(0, W, stride_w)
    for y in ys:
        axes[1].axhline(y, color='lime', lw=0.3, alpha=0.6)
    for x in xs:
        axes[1].axvline(x, color='lime', lw=0.3, alpha=0.6)
    axes[1].set_title(f"Siatka próbkowania\n(co {stride_h}×{stride_w} px)")
    axes[1].axis('off')

    axes[2].imshow(subsampled, cmap='gray')
    axes[2].set_title(f"Subsampled\n({subsampled.shape[0]}×{subsampled.shape[1]} px)")
    axes[2].axis('off')

    im = axes[3].imshow(np.log(1e-5 + power_map), cmap='hot')
    axes[3].set_title("Mapa mocy kubełków\nlog(|bucket_FFT|²)\n(jasne = aktywne kubełki)")
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# KROK 4 – Identyfikacja kandydatów z aktywnych kubełków
# ---------------------------------------------------------------------------
def show_sft_step4_candidates(cropped_image, keep_fraction=0.05, tolerance=1e-6):
    """
    Krok 4: wyszukiwanie kandydatów – aktywne kubełki (moc > próg)
    mapujemy z powrotem na konkretne współrzędne (fh, fw) w pełnym widmie.
    """
    H, W = cropped_image.shape
    k        = max(int(H * W * keep_fraction), 1)
    n_buckets = int(2 ** np.ceil(np.log2(max(4 * k, 64))))
    stride_h = max(H // n_buckets, 1)
    stride_w = max(W // n_buckets, 1)

    row_perm, col_perm, *_ = _build_permutation(H, W)
    permuted = cropped_image[np.ix_(row_perm, col_perm)]
    residual = np.fft.fft2(permuted)

    subsampled  = np.fft.ifft2(residual).real[::stride_h, ::stride_w]
    bucket_spec = np.fft.fft2(subsampled)
    power       = np.abs(bucket_spec) ** 2
    max_power   = power.max()
    threshold   = tolerance * max_power if max_power > 0 else 0.0

    active_buckets = np.argwhere(power > threshold)

    candidates_map = np.zeros((H, W))
    active_map     = np.zeros_like(power)
    active_map[active_buckets[:, 0], active_buckets[:, 1]] = 1.0

    for bh, bw in active_buckets:
        fh_start = int(bh) * stride_h
        fw_start = int(bw) * stride_w
        for fh in range(fh_start, min(fh_start + stride_h, H)):
            for fw in range(fw_start, min(fw_start + stride_w, W)):
                candidates_map[fh, fw] = np.abs(residual[fh, fw])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"SFT – Krok 4: Aktywne kubełki i kandydaci  "
        f"(próg={tolerance:.0e}, aktywnych kubełków: {len(active_buckets)})",
        fontsize=12, fontweight='bold')

    im0 = axes[0].imshow(np.log(1e-5 + power), cmap='hot')
    axes[0].set_title(f"Mapa mocy kubełków\n(aktywne = moc > próg)")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(active_map, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title(f"Aktywne kubełki\n({len(active_buckets)} z {n_buckets}×{n_buckets})")
    axes[1].axis('off')

    im2 = axes[2].imshow(np.fft.fftshift(np.log(1e-5 + candidates_map)), cmap='inferno')
    axes[2].set_title("Kandydaci – amplituda w pełnym widmie\n(wyłącznie z aktywnych kubełków)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# KROK 5 – Iteracyjna ekstrakcja składowych (ewolucja widma resztkowego)
# ---------------------------------------------------------------------------
def show_sft_step5_iterations(cropped_image, keep_fraction=0.05, tolerance=1e-6):
    """
    Krok 5: każda iteracja wyciąga top-K składowych z widma resztkowego
    i zeruje je. Zastosowano wspólną skalę kolorów (vmin, vmax), 
    aby wizualnie pokazać spadek energii resztkowej w kolejnych iteracjach.
    """
    H, W = cropped_image.shape
    k        = max(int(H * W * keep_fraction), 1)
    n_buckets = int(2 ** np.ceil(np.log2(max(4 * k, 64))))
    n_iterations = 3
    per_iter  = max(k // n_iterations, 1)
    stride_h  = max(H // n_buckets, 1)
    stride_w  = max(W // n_buckets, 1)

    row_perm, col_perm, inv_row, inv_col, *_ = _build_permutation(H, W)
    permuted  = cropped_image[np.ix_(row_perm, col_perm)]
    residual  = np.fft.fft2(permuted).copy()

    snapshots_residual = [residual.copy()]  # stan przed iteracją 0
    snapshots_recon    = []
    energy_history     = [np.sum(np.abs(residual) ** 2)]
    found_counts       = [0]

    found_freqs = {}

    for it in range(n_iterations):
        if len(found_freqs) >= k:
            break

        subsampled  = np.fft.ifft2(residual).real[::stride_h, ::stride_w]
        bucket_spec = np.fft.fft2(subsampled)
        power       = np.abs(bucket_spec) ** 2
        max_power   = power.max()
        if max_power == 0:
            break
        threshold = tolerance * max_power

        active_buckets = np.argwhere(power > threshold)
        candidates = []
        for bh, bw in active_buckets:
            fh_start = int(bh) * stride_h
            fw_start = int(bw) * stride_w
            for fh in range(fh_start, min(fh_start + stride_h, H)):
                for fw in range(fw_start, min(fw_start + stride_w, W)):
                    if (fh, fw) not in found_freqs:
                        p = abs(residual[fh, fw]) ** 2
                        candidates.append((p, fh, fw))

        candidates.sort(key=lambda x: -x[0])
        new_count = 0
        for p, fh, fw in candidates:
            if len(found_freqs) >= k or new_count >= per_iter:
                break
            if (fh, fw) in found_freqs:
                continue
            found_freqs[(fh, fw)] = residual[fh, fw]
            residual[fh, fw] = 0.0
            new_count += 1

        snapshots_residual.append(residual.copy())
        energy_history.append(np.sum(np.abs(residual) ** 2))
        found_counts.append(len(found_freqs))

        # Rekonstrukcja na tym etapie
        partial_spec = np.zeros((H, W), dtype=complex)
        for (fh, fw), val in found_freqs.items():
            partial_spec[fh, fw] = val
        part_rec = np.fft.ifft2(partial_spec).real
        part_rec_unpermed = part_rec[np.ix_(inv_row, inv_col)]
        snapshots_recon.append(np.clip(part_rec_unpermed, 0, 255))

    n_snaps = len(snapshots_recon)
    fig, axes = plt.subplots(3, n_snaps + 1, figsize=(4 * (n_snaps + 1), 11))
    fig.suptitle(
        f"SFT – Krok 5: Iteracyjna ekstrakcja  "
        f"(k={k}, {n_iterations} iteracje, ~{per_iter} wsp. na iterację)",
        fontsize=12, fontweight='bold')

    # Poprawka: Obliczenie globalnego minimum i maksimum dla log-widma, 
    # aby widoczny był fizyczny spadek energii (zjawisko blaknięcia wykresu).
    base_log_spec = _log_spectrum(snapshots_residual[0])
    vmax_spec = base_log_spec.max()
    vmin_spec = base_log_spec.min()

    # Wiersz 0: widmo resztkowe
    for i, snap in enumerate(snapshots_residual[:n_snaps + 1]):
        lbl = "Przed iter. 1" if i == 0 else f"Po iteracji {i}"
        im = axes[0, i].imshow(_log_spectrum(snap), cmap='inferno', vmin=vmin_spec, vmax=vmax_spec)
        axes[0, i].set_title(lbl, fontsize=9)
        axes[0, i].axis('off')
        axes[0, i].set_xlabel(f"Energia: {np.sum(np.abs(snap)**2):.2e}", fontsize=7)
    axes[0, 0].set_ylabel("Widmo resztkowe", fontsize=9, labelpad=4)

    # Wiersz 1: częściowa rekonstrukcja
    axes[1, 0].imshow(cropped_image, cmap='gray')
    axes[1, 0].set_title("Oryginał", fontsize=9)
    axes[1, 0].axis('off')
    for i, recon in enumerate(snapshots_recon):
        im = axes[1, i + 1].imshow(recon, cmap='gray')
        axes[1, i + 1].set_title(f"Rekonstr. po iter. {i+1}\n({found_counts[i+1]} wsp.)", fontsize=9)
        axes[1, i + 1].axis('off')
    axes[1, 0].set_ylabel("Rekonstrukcja", fontsize=9, labelpad=4)

    # Wiersz 2: wykres spadku energii resztkowej
    ax_e = axes[2, 0]
    iters = range(len(energy_history))
    ax_e.plot(iters, energy_history, 'o-', color='tab:red', linewidth=2, markersize=6)
    ax_e.set_title("Energia resztkowa vs. iteracja", fontsize=9)
    ax_e.set_xlabel("Iteracja")
    ax_e.set_ylabel("Σ|R|²")
    ax_e.grid(alpha=0.3)
    ax_e.spines[['top', 'right']].set_visible(False)

    ax_k = axes[2, 1]
    ax_k.bar(range(1, len(found_counts)), found_counts[1:],
             color='tab:blue', alpha=0.8)
    ax_k.set_title("Znalezione współczynniki vs. iteracja", fontsize=9)
    ax_k.set_xlabel("Iteracja")
    ax_k.set_ylabel("Łączna liczba wsp.")
    ax_k.axhline(k, color='red', linestyle='--', label=f'cel k={k}')
    ax_k.legend(fontsize=8)
    ax_k.grid(alpha=0.3)
    ax_k.spines[['top', 'right']].set_visible(False)

    for col in range(2, n_snaps + 1):
        axes[2, col].axis('off')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# KROK 6 – Sparse spectrum vs pełne FFT
# ---------------------------------------------------------------------------
def show_sft_step6_sparse_spectrum(cropped_image, keep_fraction=0.05):
    """
    Krok 6: końcowe sparse widmo – tylko k wybranych współczynników ≠ 0,
    reszta wyzerowana. Porównanie z pełnym FFT.
    """
    H, W = cropped_image.shape
    k = max(int(H * W * keep_fraction), 1)

    row_perm, col_perm, *_ = _build_permutation(H, W)
    permuted = cropped_image[np.ix_(row_perm, col_perm)]
    full_spectrum = np.fft.fft2(permuted)

    # Poprawka: Zastosowano np.partition dla znacznie szybszego wydzielenia 
    # k-tego największego elementu zamiast sortowania całej tablicy (złożoność O(N) zamiast O(N log N)).
    flat_amp = np.abs(full_spectrum).flatten()
    thresh   = np.partition(flat_amp, -k)[-k]
    sparse   = np.where(np.abs(full_spectrum) >= thresh, full_spectrum, 0)

    total_energy  = np.sum(np.abs(full_spectrum) ** 2)
    sparse_energy = np.sum(np.abs(sparse) ** 2)
    kept_pct      = 100 * sparse_energy / total_energy if total_energy > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"SFT – Krok 6: Sparse spectrum  "
        f"({k} wsp. = {keep_fraction*100:.2f}% — zachowana energia: {kept_pct:.1f}%)",
        fontsize=12, fontweight='bold')

    im0 = axes[0].imshow(_log_spectrum(full_spectrum), cmap='inferno')
    axes[0].set_title(f"Pełne FFT\n({H*W} współczynników)")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(_log_spectrum(sparse), cmap='inferno')
    axes[1].set_title(f"Sparse spectrum (SFT)\n({k} współczynników ≠ 0)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Maska: które punkty widma zostały zachowane
    mask = (np.abs(sparse) > 0).astype(float)
    im2 = axes[2].imshow(np.fft.fftshift(mask), cmap='binary', vmin=0, vmax=1)
    axes[2].set_title(f"Maska zachowanych wsp.\n(białe = zachowane, {k}/{H*W})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# KROK 7 – Odwrotna permutacja i końcowa rekonstrukcja
# ---------------------------------------------------------------------------
def show_sft_step7_reconstruction(cropped_image, keep_fraction=0.05):
    """
    Krok 7: IFFT2 sparse spectrum → obraz w przestrzeni permutowanej,
    następnie odwrotna permutacja → finalny zrekonstruowany obraz.
    Porównanie z oryginałem + mapa błędu.
    """
    H, W = cropped_image.shape
    k = max(int(H * W * keep_fraction), 1)

    row_perm, col_perm, inv_row, inv_col, *_ = _build_permutation(H, W)
    permuted      = cropped_image[np.ix_(row_perm, col_perm)]
    full_spectrum = np.fft.fft2(permuted)

    # Poprawka: np.partition (zgodnie ze standardami wydajności dydaktycznej NumPy)
    flat_amp = np.abs(full_spectrum).flatten()
    thresh   = np.partition(flat_amp, -k)[-k]
    sparse   = np.where(np.abs(full_spectrum) >= thresh, full_spectrum, 0)

    permuted_recon = np.fft.ifft2(sparse).real
    reconstructed  = np.clip(permuted_recon[np.ix_(inv_row, inv_col)], 0, 255)

    orig_f  = cropped_image.astype(float)
    recon_f = reconstructed.astype(float)
    error   = np.abs(orig_f - recon_f)

    from skimage.metrics import structural_similarity as ssim_metric
    data_range = 1.0 if orig_f.max() <= 1.0 else 255.0
    mse  = np.mean(error ** 2)
    psnr = 10 * np.log10(data_range ** 2 / mse) if mse > 0 else float('inf')
    ssim = ssim_metric(orig_f, recon_f, data_range=data_range)
    snr  = 10 * np.log10(np.sum(orig_f**2) / np.sum(error**2))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"SFT – Krok 7: Odwrotna permutacja i rekonstrukcja  "
        f"(k={k}, keep={keep_fraction*100:.2f}%)",
        fontsize=12, fontweight='bold')

    axes[0, 0].imshow(permuted, cmap='gray')
    axes[0, 0].set_title("Spermutowany obraz (wejście FFT)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(permuted_recon, cmap='gray')
    axes[0, 1].set_title("IFFT sparse spectrum\n(w przestrzeni permutowanej)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(reconstructed, cmap='gray')
    axes[0, 2].set_title("Po odwrotnej permutacji\n(finalny wynik SFT)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(cropped_image, cmap='gray')
    axes[1, 0].set_title("Oryginał")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(reconstructed, cmap='gray')
    axes[1, 1].set_title(
        f"Rekonstrukcja SFT\nSNR={snr:.2f} dB | PSNR={psnr:.2f} dB | SSIM={ssim:.4f}")
    axes[1, 1].axis('off')

    im = axes[1, 2].imshow(error, cmap='hot', vmin=0)
    axes[1, 2].set_title(f"Mapa błędu |oryginał − rekonstr.|\nMSE={mse:.4f}")
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# BONUS – Wpływ keep_fraction na jakość rekonstrukcji
# ---------------------------------------------------------------------------
def show_sft_keep_fraction_sweep(cropped_image,
                                  fractions=(0.001, 0.005, 0.01, 0.05, 0.1, 0.2)):
    """
    Bonusowa wizualizacja: jak zmienia się jakość rekonstrukcji
    w zależności od parametru keep_fraction (= ile % współczynników zachowujemy).
    """
    from algorithms.sft import sft  # importujemy oryginalne sft

    H, W = cropped_image.shape
    orig_f = cropped_image.astype(float)
    data_range = 1.0 if orig_f.max() <= 1.0 else 255.0

    results = []
    for frac in fractions:
        recon = sft(cropped_image, keep_fraction=frac)
        recon_f = recon.astype(float)
        mse  = np.mean((orig_f - recon_f) ** 2)
        snr  = 10 * np.log10(np.sum(orig_f**2) / np.sum((orig_f-recon_f)**2))
        psnr = 10 * np.log10(data_range**2 / mse) if mse > 0 else float('inf')
        k    = max(int(H * W * frac), 1)
        results.append({'frac': frac, 'k': k, 'recon': recon, 'snr': snr,
                        'psnr': psnr, 'mse': mse})

    n = len(fractions)
    fig, axes = plt.subplots(3, n, figsize=(3.5 * n, 10))
    fig.suptitle("SFT – Wpływ parametru keep_fraction na jakość rekonstrukcji",
                 fontsize=13, fontweight='bold')

    for i, r in enumerate(results):
        axes[0, i].imshow(r['recon'], cmap='gray')
        axes[0, i].set_title(
            f"frac={r['frac']}\nk={r['k']}", fontsize=8)
        axes[0, i].axis('off')

        error = np.abs(orig_f - r['recon'].astype(float))
        axes[1, i].imshow(error, cmap='hot', vmin=0)
        axes[1, i].set_title(f"MSE={r['mse']:.4f}", fontsize=8)
        axes[1, i].axis('off')

    # Wiersz 2: wykresy SNR i PSNR
    fracs   = [r['frac'] * 100 for r in results]
    snrs    = [r['snr']  for r in results]
    psnrs   = [r['psnr'] for r in results]

    ax_snr = axes[2, 0]
    ax_snr.plot(fracs, snrs,  'o-', color='tab:blue',  label='SNR')
    ax_snr.plot(fracs, psnrs, 's-', color='tab:orange', label='PSNR')
    ax_snr.set_xlabel("keep_fraction (%)")
    ax_snr.set_ylabel("dB")
    ax_snr.set_title("SNR i PSNR vs keep_fraction")
    ax_snr.legend(fontsize=8)
    ax_snr.grid(alpha=0.3)
    ax_snr.spines[['top', 'right']].set_visible(False)

    for col in range(1, n):
        axes[2, col].axis('off')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Główna funkcja uruchamiająca wszystkie kroki naraz
# ---------------------------------------------------------------------------
def show_sft_steps(cropped_image, keep_fraction=0.05):
    """
    Uruchamia wizualizacje wszystkich 7 kroków algorytmu SFT:

      Krok 1 – Losowa permutacja wierszy i kolumn
      Krok 2 – Pełne FFT2 spermutowanego sygnału (widmo resztkowe)
      Krok 3 – Haszowanie sygnału do kubełków (subsampling + lokalne FFT)
      Krok 4 – Identyfikacja kandydatów z aktywnych kubełków
      Krok 5 – Iteracyjna ekstrakcja składowych i ewolucja widma resztkowego
      Krok 6 – Finalne sparse spectrum vs pełne FFT
      Krok 7 – Odwrotna permutacja i rekonstrukcja obrazu
    """
    print("=== SFT – wizualizacja krok po kroku ===")
    print(f"Obraz: {cropped_image.shape}, keep_fraction={keep_fraction}")
    k = max(int(cropped_image.shape[0] * cropped_image.shape[1] * keep_fraction), 1)
    print(f"Liczba zachowanych współczynników k={k} "
          f"({keep_fraction*100:.2f}% z {cropped_image.size})\n")

    print("Krok 1: Permutacja...")
    show_sft_step1_permutation(cropped_image)

    print("Krok 2: Pełne FFT2 (widmo resztkowe)...")
    show_sft_step2_full_spectrum(cropped_image)

    print("Krok 3: Haszowanie do kubełków...")
    show_sft_step3_bucket_hashing(cropped_image, keep_fraction)

    print("Krok 4: Kandydaci z aktywnych kubełków...")
    show_sft_step4_candidates(cropped_image, keep_fraction)

    print("Krok 5: Iteracyjna ekstrakcja...")
    show_sft_step5_iterations(cropped_image, keep_fraction)

    print("Krok 6: Sparse spectrum...")
    show_sft_step6_sparse_spectrum(cropped_image, keep_fraction)

    print("Krok 7: Rekonstrukcja...")
    show_sft_step7_reconstruction(cropped_image, keep_fraction)

    print("=== Wizualizacja zakończona ===")