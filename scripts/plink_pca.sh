NTHREADS=32

mkdir -p pca

# Convert to plink binary; --double-id uses the VCF sample ID for both FID and IID
plink --vcf raw/db.vcf.gz \
      --threads $NTHREADS \
      --make-bed \
      --double-id \
      --out pca/db

# LD pruning: 50-SNP window, step 10, r2 threshold 0.1
plink --bfile pca/db \
      --threads $NTHREADS \
      --indep-pairwise 50 10 0.1 \
      --out pca/db

# PCA on pruned SNPs, keep top 20 PCs, output variant weights as well
plink --bfile pca/db \
      --threads $NTHREADS \
      --extract pca/db.prune.in \
      --pca 20 var-wts \
      --out pca/db_pca
