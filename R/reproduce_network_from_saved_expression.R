#!/usr/bin/env Rscript
# Replay corrected associations/display using published expression and fixed module assignments.
suppressPackageStartupMessages(library(WGCNA))
args <- commandArgs(trailingOnly=TRUE)
root <- if (length(args)>0) normalizePath(args[1]) else normalizePath('.')
out <- if (length(args)>1) args[2] else file.path(root,'outputs','WGCNA_replay')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
e <- new.env(parent=emptyenv()); load(file.path(root,'R','cleaned_data_for_WGCNA.RData'),envir=e)
X <- as.matrix(e$datExpr); storage.mode(X) <- 'double'
stats <- read.csv(file.path(root,'results','revised','WGCNA','gene_statistics_signed_and_absolute.csv'),check.names=FALSE)
manifest <- read.csv(file.path(root,'results','revised','WGCNA','sample_manifest_n23.csv'),check.names=FALSE)
stopifnot(nrow(X)==23,ncol(X)==27910,!anyDuplicated(colnames(X)),!anyDuplicated(rownames(X)))
manifest <- manifest[match(rownames(X),manifest$Samples),]; stopifnot(!anyNA(manifest$Samples))
L <- manifest$Leaf_indicator; stopifnot(sum(L==1)==14,sum(L==0)==9)
stats <- stats[match(colnames(X),stats$gene_id),]; stopifnot(!anyNA(stats$gene_id))
colors <- stats$module; names(colors) <- stats$gene_id
me <- moduleEigengenes(X,colors=colors,align='along average')$eigengenes
leaf.cor <- as.numeric(cor(me,L)); assoc <- data.frame(module=sub('^ME','',names(me)),r_leaf=leaf.cor,r_root=-leaf.cor,p=corPvalueStudent(leaf.cor,23))
assoc$BH_assigned <- NA_real_; assigned <- assoc$module!='grey'; assoc$BH_assigned[assigned] <- p.adjust(assoc$p[assigned],'BH')
write.csv(assoc,file.path(out,'module_associations.csv'),row.names=FALSE)
# Original-module connectivity ranking is a supplied numerical input, retained to reproduce the displayed selection.
rank <- read.csv(file.path(root,'results','revised','WGCNA','two_modules_with_original_kWithin.csv'),check.names=FALSE)
kcol <- grep('kWithin|kWithin_original',names(rank),value=TRUE)[1]
if (is.na(kcol)) stop('No original intramodular connectivity column')
rank <- rank[order(-rank[[kcol]],rank$gene_id),]; selected <- head(rank$gene_id,500)
stopifnot(length(selected)==500,all(selected %in% colnames(X)))
# A_selected_all carries full-network neighbor contributions without allocating a 27910-square matrix.
A <- abs(cor(X[,selected,drop=FALSE],X))^12
A[cbind(seq_along(selected),match(selected,colnames(X)))] <- 0
ki <- rowSums(A); aa <- A[,match(selected,colnames(X)),drop=FALSE]
TOM <- (tcrossprod(A)+aa)/(outer(ki,ki,pmin)+1-aa);diag(TOM) <- 0
threshold <- as.numeric(quantile(TOM[upper.tri(TOM)],.95,type=7))
idx <- which(upper.tri(TOM)&TOM>threshold,arr.ind=TRUE)
edges <- data.frame(from=selected[idx[,1]],to=selected[idx[,2]],TOM=TOM[idx])
degree <- tabulate(c(idx[,1],idx[,2]),nbins=500);nodes <- data.frame(gene_id=selected,module=colors[selected],degree=degree)
hubs <- nodes[nodes$degree>=100,]; induced <- edges[edges$from %in% hubs$gene_id & edges$to %in% hubs$gene_id,]
write.csv(edges,file.path(out,'display_edges.csv'),row.names=FALSE);write.csv(nodes,file.path(out,'display_nodes.csv'),row.names=FALSE)
write.csv(hubs,file.path(out,'hubs.csv'),row.names=FALSE);write.csv(induced,file.path(out,'hub_induced_edges.csv'),row.names=FALSE)
write.csv(data.frame(TOM_threshold=threshold,n_edges=nrow(edges),n_nonisolated=sum(degree>0),n_hubs=nrow(hubs),hub_edges=nrow(induced)),file.path(out,'counts.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'sessionInfo.txt'))
