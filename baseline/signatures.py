def get_signature(signature):
    prostadiag = sorted(['rna_CD38','rna_NCAPD3','rna_AZGP1','rna_AFF3','rna_SLC22A3',
                         'rna_FMOD','rna_CHRNA2','rna_SLC15A2','rna_ACADL','rna_GREB1',
                         'rna_ANPEP','rna_ANO7','rna_REPS2','rna_HGD','rna_PAK1IP1',
                         'rna_COL1A2','rna_SPARC','rna_MS4A6A','rna_FRZB','rna_MGP',
                         'rna_COL3A1','rna_CDH11','rna_MOXD1','rna_COL8A1','rna_COL1A1',
                         'rna_NOX4','rna_THBS2','rna_VCAN','rna_SULF1','rna_SFRP2',
                         'rna_KHDRBS3','rna_COL10A1','rna_CXCL14','rna_SFRP4','rna_ASPN','rna_COMP'])

    decipher = sorted(['rna_LASP1', 'rna_IQGAP3', 'rna_NFIB', 'rna_S1RP4', 'rna_THBS2', 'rna_ANO7', 
                       'rna_PCDH7','rna_MYBPC1','rna_EPPK1','rna_TSBP', 'rna_PBX1','rna_NUSAP1', 
                       'rna_ZWILCH', 'rna_UBE2C','rna_CAMK2N1','rna_RABGAP1', 'rna_PCAT-32', 
                       'rna_GLYATL1P4', 'rna_PCAT-80', 'rna_TNFRSF19'])

    oncotype_dx = sorted(['rna_SFRP4', 'rna_BGN', 'rna_COL1A1', 'rna_KLK2', 'rna_SRD5A2', 'rna_FAM13C',
                          'rna_AZGP1','rna_GSN', 'rna_GSTM2', 'rna_TPM2', 'rna_FLNC', 'rna_TPX2'])

    if (signature == "prostadiag"):
        return prostadiag
    elif (signature == "decipher"):
        return decipher
    elif (signature == "oncotype_dx"):
        return oncotype_dx