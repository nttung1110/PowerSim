// Exact power-diagram adjacency via Geogram's RegularWeightedDelaunay3d ("BPOW").
//
// Why: scene_prep.build_exact_power_adjacency uses scipy = Qhull's general-dimension convex hull
// in 4D, which the paragram paper's Table 2 measures at 230.98 s on 4.1M points where Geogram
// takes 2.67 s -- an 86x penalty that comes from the ALGORITHM CHOICE, not from exactness. This
// shim reaches Geogram's purpose-built regular triangulation directly; the pip `geogram` binding
// only exposes the RESTRICTED diagram (949k tets for 20k seeds), which is a much heavier path.
//
// I/O is a raw binary file so no build-time Python coupling is needed:
//   in : int64 n | n*3 float64 xyz | n float64 weights
//   out: int64 n, int64 n_edges | (n+1) int64 CSR offsets | n_edges int32 neighbours
//
// The weight sign convention is passed in by the caller and settled EMPIRICALLY against scipy --
// geogram's internal lift negates and squares in a way that is easy to get backwards, and a wrong
// sign yields a valid-but-different diagram rather than an error.
#include "Delaunay_psm.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <set>
#include <cmath>
#include <cstdlib>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s in.bin out.bin [method=BPOW|PDEL] [threads]\n", argv[0]); return 1; }
    const char* method = (argc > 3) ? argv[3] : "BPOW";
    GEO::initialize();
    // Geogram does not multithread unless told to. "BPOW" (RegularWeightedDelaunay3d) is serial;
    // "PDEL" (ParallelDelaunay3d) accepts dimension 3 OR 4 and carries weighted_/heights_, so the
    // parallel path handles the weighted case too. Threads default to all cores; override with
    // the 4th argv.
    GEO::Process::enable_multithreading(true);
    if (argc > 4) {
        GEO::Process::set_max_threads(GEO::index_t(atoi(argv[4])));
    }
    fprintf(stderr, "threads: %u\n", GEO::Process::maximum_concurrent_threads());
    // GEO_BENCH=1 turns on geogram's own internal phase timers (BRIO spatial sort vs insertion),
    // which is how we find out whether the per-frame spatial reorder is worth eliminating.
    if (const char* b = getenv("GEO_BENCH")) {
        if (b[0] == '1') GEO::CmdLine::set_arg("dbg:delaunay_benchmark", true);
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    int64_t n = 0;
    if (fread(&n, sizeof(int64_t), 1, f) != 1) return 1;
    std::vector<double> xyz(3 * n), w(n);
    if (fread(xyz.data(), sizeof(double), 3 * n, f) != size_t(3 * n)) return 1;
    if (fread(w.data(), sizeof(double), n, f) != size_t(n)) return 1;
    fclose(f);

    // BPOW's 4th coordinate is NOT the weight. Delaunay3d::set_vertices documents the contract:
    //     "Client code uses 4d embedding with ti = sqrt(W - wi) where W = max(wi)"
    // and then recovers w internally as -(ti^2). Passing the weight directly is silently wrong --
    // because it SQUARES the 4th coordinate, +r^2 and -r^2 give identical results, which is how
    // the mistake shows up: the diagram comes out unweighted rather than erroring.
    double W = *std::max_element(w.begin(), w.end());
    std::vector<double> pts4(4 * n);
    for (int64_t i = 0; i < n; ++i) {
        pts4[4*i+0] = xyz[3*i+0]; pts4[4*i+1] = xyz[3*i+1]; pts4[4*i+2] = xyz[3*i+2];
        double d = W - w[i];
        pts4[4*i+3] = std::sqrt(d > 0.0 ? d : 0.0);
    }

    auto T0 = std::chrono::steady_clock::now();
    GEO::Delaunay_var del = GEO::Delaunay::create(4, method);
    // GEO_NOREORDER=1 skips geogram's per-frame BRIO spatial sort. Only sensible if the caller
    // feeds spatially-coherent input -- on arbitrary ordering it makes insertion much slower,
    // because BRIO exists to give the point-location walks locality.
    if (const char* nr = getenv("GEO_NOREORDER")) {
        if (nr[0] == '1') del->set_reorder(false);
    }
    del->set_vertices(GEO::index_t(n), pts4.data());
    const GEO::index_t nc = del->nb_cells();
    auto T1 = std::chrono::steady_clock::now();
    fprintf(stderr, "geogram: %s, %lld sites -> %u cells\n", method, (long long)n, nc);

    // Adjacency = edges of the regular triangulation (its dual is the power diagram).
    //
    // Two earlier versions were the bottleneck, both measured: 1.2M std::sets (18M insertions,
    // each an allocation plus a tree walk) and then a single sort+unique over ~94M pairs (~750 MB,
    // 8.7 s -- 85% of total runtime while the triangulation itself took only 1.58 s).
    //
    // This bucketizes instead: count per-row degrees, prefix-sum, scatter, then sort+unique each
    // row independently. Rows are ~90 entries with duplicates (each edge is shared by ~6 tets)
    // collapsing to ~15, so the per-row sorts are tiny and cache-resident -- and independent, so
    // they parallelise cleanly.
    std::vector<int64_t> cnt(n + 1, 0);
    for (GEO::index_t c = 0; c < nc; ++c) {
        GEO::signed_index_t v[4];
        for (int k = 0; k < 4; ++k) v[k] = del->cell_vertex(c, GEO::index_t(k));
        for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b) {
            if (v[a] < 0 || v[b] < 0) continue;
            cnt[v[a] + 1]++; cnt[v[b] + 1]++;
        }
    }
    std::vector<int64_t> start(n + 1, 0);
    for (int64_t i = 0; i < n; ++i) start[i + 1] = start[i] + cnt[i + 1];
    std::vector<int32_t> raw(size_t(start[n]));
    std::vector<int64_t> cur(start.begin(), start.end() - 1);
    for (GEO::index_t c = 0; c < nc; ++c) {
        GEO::signed_index_t v[4];
        for (int k = 0; k < 4; ++k) v[k] = del->cell_vertex(c, GEO::index_t(k));
        for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b) {
            if (v[a] < 0 || v[b] < 0) continue;
            raw[size_t(cur[v[a]]++)] = int32_t(v[b]);
            raw[size_t(cur[v[b]]++)] = int32_t(v[a]);
        }
    }

    std::vector<int32_t> uniq_len(n, 0);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        int32_t* b = raw.data() + start[i];
        int32_t* e = raw.data() + start[i + 1];
        std::sort(b, e);
        uniq_len[i] = int32_t(std::unique(b, e) - b);
    }

    std::vector<int64_t> off(n + 1, 0);
    for (int64_t i = 0; i < n; ++i) off[i + 1] = off[i] + uniq_len[i];
    std::vector<int32_t> flat(size_t(off[n]));
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        std::copy(raw.data() + start[i], raw.data() + start[i] + uniq_len[i],
                  flat.data() + off[i]);
    }
    auto T2 = std::chrono::steady_clock::now();
    fprintf(stderr, "  timing: triangulation %.2fs, adjacency %.2fs\n",
            std::chrono::duration<double>(T1-T0).count(),
            std::chrono::duration<double>(T2-T1).count());

    FILE* g = fopen(argv[2], "wb");
    int64_t ne = off[n];
    fwrite(&n, sizeof(int64_t), 1, g);
    fwrite(&ne, sizeof(int64_t), 1, g);
    fwrite(off.data(), sizeof(int64_t), size_t(n + 1), g);
    fwrite(flat.data(), sizeof(int32_t), size_t(ne), g);
    fclose(g);
    fprintf(stderr, "wrote %lld edges\n", (long long)ne);
    return 0;
}
