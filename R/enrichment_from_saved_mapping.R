#!/usr/bin/env Rscript
# Default reports supplied frozen exports. --rerun executes a new annotation query and records its date.
args <- commandArgs(trailingOnly=TRUE)
root <- if(length(args)>0) normalizePath(args[1]) else normalizePath('.')
folder <- file.path(root,'results','revised','WGCNA','enrichment_modules')
if (!('--rerun' %in% args)) {
 for (n in c('Fig5_GO_BP_FDR05.csv','Fig5_GO_CC_FDR05.csv','Fig5_GO_MF_FDR05.csv','Fig5_KEGG_FDR05.csv')) {
  x <- read.csv(file.path(folder,n));cat(n,':',nrow(x),'significant terms\n')
 }
 quit(save='no')
}
suppressPackageStartupMessages(library(clusterProfiler));suppressPackageStartupMessages(library(org.Zmays.eg.db))
out <- file.path(root,'outputs','enrichment_new_query');dir.create(out,recursive=TRUE,showWarnings=FALSE)
genes <- scan(file.path(folder,'target_unique_Entrez_IDs.txt'),what=character(),quiet=TRUE)
universe <- scan(file.path(folder,'background_unique_Entrez_IDs.txt'),what=character(),quiet=TRUE)
for (ont in c('BP','CC','MF')) {
 z <- enrichGO(unique(genes),OrgDb=org.Zmays.eg.db,keyType='ENTREZID',ont=ont,universe=unique(universe),pAdjustMethod='BH',pvalueCutoff=1,qvalueCutoff=1,minGSSize=10,maxGSSize=500,readable=FALSE)
 saveRDS(z,file.path(out,paste0('GO_',ont,'.rds')));x <- as.data.frame(z);write.csv(x,file.path(out,paste0('GO_',ont,'_all_tested.csv')),row.names=FALSE);write.csv(x[x$p.adjust<.05,,drop=FALSE],file.path(out,paste0('GO_',ont,'_FDR05.csv')),row.names=FALSE)
}
z <- enrichKEGG(unique(genes),organism='zma',keyType='ncbi-geneid',universe=unique(universe),pAdjustMethod='BH',pvalueCutoff=1,qvalueCutoff=1,minGSSize=10,maxGSSize=500)
saveRDS(z,file.path(out,'KEGG.rds'));x <- as.data.frame(z);write.csv(x,file.path(out,'KEGG_all_tested.csv'),row.names=FALSE);write.csv(x[x$p.adjust<.05,,drop=FALSE],file.path(out,'KEGG_FDR05.csv'),row.names=FALSE)
writeLines(c(paste('New annotation query date',Sys.Date()),capture.output(sessionInfo())),file.path(out,'provenance.txt'))
