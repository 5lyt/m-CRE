import gffutils
from Bio import SeqIO
import pandas as pd
import re


def extract_gene_id(attributes):
    """提取玉米基因ID的优先顺序"""
    patterns = [
        r'gene:(Zm\w+)',  # 匹配ID字段如 ID=gene:Zm00001d027
        r'gene=(Zm\w+)',  # 匹配gene=Zm00001d027
        r'Name=(Zm\w+)',  # 匹配Name字段
        r'ID=(Zm\w+)[.;]',  # 匹配ID字段
        r'locus_tag=(Zm\w+)'  # 备用匹配
    ]

    for pattern in patterns:
        match = re.search(pattern, attributes)
        if match:
            return match.group(1)
    return None


def calculate_gc_content(sequence):
    """计算GC含量"""
    if not sequence:
        return 0.0
    gc_count = sequence.count("G") + sequence.count("C")
    return gc_count / len(sequence)


def extract_mrna_features(gff_file, fasta_file):
    # 创建内存数据库
    db = gffutils.create_db(
        gff_file, dbfn=":memory:",
        force=True, keep_order=True,
        merge_strategy="merge",
        sort_attribute_values=True
    )

    fasta = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    results = []

    for mrna in db.features_of_type("mRNA"):
        # 组合属性字段
        attrs = ";".join([f"{k}={','.join(v)}" for k, v in mrna.attributes.items()])
        gene_id = extract_gene_id(attrs)

        if not gene_id:
            continue

        try:
            seq_record = fasta[mrna.seqid]
            sequence_str = str(seq_record.seq)
            seq_obj = seq_record.seq

            # 初始化特征值
            utr5_len = utr3_len = cds_len = intron_len = 0
            utr5_gc = utr3_gc = cds_gc = 0.0
            cds_total = 0

            # 处理CDS区域
            cds_regions = sorted(
                [(cds.start, cds.end) for cds in db.children(mrna, featuretype="CDS")],
                key=lambda x: x[0]
            )

            if cds_regions:
                first_cds_start = cds_regions[0][0]
                last_cds_end = cds_regions[-1][1]

                # 处理UTR序列
                if mrna.strand == "+":
                    utr5_seq = sequence_str[mrna.start - 1: first_cds_start - 1]
                    utr3_seq = sequence_str[last_cds_end: mrna.end]
                else:
                    # 获取反向互补序列
                    utr5_subseq = seq_obj[last_cds_end: mrna.end]
                    utr5_seq = str(utr5_subseq.reverse_complement())
                    utr3_subseq = seq_obj[mrna.start - 1: first_cds_start - 1]
                    utr3_seq = str(utr3_subseq.reverse_complement())

                utr5_len = len(utr5_seq)
                utr5_gc = calculate_gc_content(utr5_seq)
                utr3_len = len(utr3_seq)
                utr3_gc = calculate_gc_content(utr3_seq)

                # 计算CDS总长和GC
                cds_total = sum(end - start + 1 for start, end in cds_regions)
                cds_seq = "".join(sequence_str[start - 1:end] for start, end in cds_regions)
                cds_gc = calculate_gc_content(cds_seq)

            # 计算内含子长度
            exons = list(db.children(mrna, featuretype="exon"))
            exon_total = sum(exon.end - exon.start + 1 for exon in exons)
            intron_len = (mrna.end - mrna.start + 1) - exon_total

            # 计算ORF外显子密度
            orf_exon_density = len(exons) / cds_total if cds_total > 0 else 0

            # 保存结果
            results.append({
                "Gene_ID": gene_id,
                "UTR5LEN": utr5_len,
                "CDSLEN": cds_total,
                "INTRONLEN": intron_len,
                "UTR3LEN": utr3_len,
                "UTR5GC": utr5_gc,
                "CDSGC": cds_gc,
                "UTR3GC": utr3_gc,
                "ORFEXONDENSITY": orf_exon_density
            })

        except Exception as e:
            print(f"Error processing {gene_id}: {str(e)}")

    return pd.DataFrame(results)


def main():
    fna_path = "/data/Zea_mays/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa"
    gff_path = "/data/Zea_mays/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.60.gff3/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.60.gff3"

    df = extract_mrna_features(gff_path, fna_path)

    # 加权GC计算函数
    def weighted_gc(group):
        total_length = group['CDSLEN'].sum()
        group['CDSGC_WEIGHTED'] = group['CDSGC'] * group['CDSLEN']
        return group['CDSGC_WEIGHTED'].sum() / total_length if total_length > 0 else 0

    # 分组聚合
    aggregations = {
        'UTR5LEN': 'max',
        'CDSLEN': 'max',
        'INTRONLEN': 'sum',
        'UTR3LEN': 'max',
        'ORFEXONDENSITY': 'mean'
    }

    # 分组计算
    df_grouped = df.groupby('Gene_ID').agg(aggregations)

    # 计算加权GC
    for gc_col in ['UTR5GC', 'CDSGC', 'UTR3GC']:
        df_grouped[gc_col] = df.groupby('Gene_ID').apply(
            lambda g: (g[gc_col] * g[gc_col.replace('GC', 'LEN')]).sum() / g[gc_col.replace('GC', 'LEN')].sum()
        )

    df_grouped.reset_index(inplace=True)

    # 保存完整结果
    df.to_csv("Zmay_mRNA_features.csv", index=False)


if __name__ == "__main__":
    main()