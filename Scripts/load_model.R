library(glmnet)

cvfit = readRDS('/disk2/user/manche/Panacea/data/external_rds_files/cvfit_final.rds')
head(cvfit)
rownames(cvfit$glmnet.fit$beta) %>% head
v_model_genes = rownames(cvfit$glmnet.fit$beta)

#Read ranked file in csv format
df <- read.csv("/disk2/user/manche/Panacea/output/vae/drug_responses_a549_ranked_ens.csv", header=TRUE)

# Creating a subset with only three drugs
subset_df <- subset(df, select = c("ensembl_id", "afatinib", "erlotinib", "neratinib"))

mat = as.matrix(subset_df[,-1])
head(mat)
length(mat)
predict(cvfit, t(mat))
