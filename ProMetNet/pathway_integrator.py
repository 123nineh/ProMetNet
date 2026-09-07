import pandas as pd
from pathlib import Path

class PathwayDataIntegrator:
    """
      Integrate proteomics and metabolomics data by aligning samples and extracting pathway-matched features.

      This class aligns proteomic and metabolomic datasets based on shared pathways,
      extracts the corresponding expression data, and merges sample group information
      for downstream multi-omics analyses.

      Args:
          prot_data (pd.DataFrame): Proteomics data with features in the first column.
          meta_data (pd.DataFrame): Metabolomics data with features in the first column.
          prot_group_data (pd.DataFrame): Proteomics sample group info.
          meta_group_data (pd.DataFrame): Metabolomics sample group info.
          translation_data (pd.DataFrame): Mapping of features to common pathways.
          output_dir (Path or str): Directory for saving outputs.

      Attributes:
          expression_matrix (pd.DataFrame): Combined expression data for common pathways.
          group_matrix (pd.DataFrame): Merged sample group information.
      """
    def __init__(self, prot_data, meta_data, prot_group_data, meta_group_data, translation_data, output_dir):
        self.prot_df = prot_data
        self.meta_df = meta_data
        self.group1 = prot_group_data
        self.group2 = meta_group_data
        self.translation = translation_data
        self.sample = None
        self.df1_processed = None
        self.df2_processed = None
        self.pathways = None
        self.output_dir = output_dir
        self.expression_matrix = None
        self.group_matrix = None

    def align_samples(self):
        s1 = pd.DataFrame(self.prot_df.columns[1:], columns=['sample'])
        s2 = pd.DataFrame(self.meta_df.columns[1:], columns=['sample'])
        self.sample = pd.merge(s1, s2, on='sample')
        self.df1_processed = self.prot_df[['Features'] + self.sample['sample'].tolist()]
        self.df2_processed = self.meta_df[['Features'] + self.sample['sample'].tolist()]

    def filter_common_pathways(self):
        prot_input = self.df1_processed['Features'].to_frame(name='input')
        meta_input = self.df2_processed['Features'].to_frame(name='input')
        prot_pathways = pd.merge(prot_input, self.translation, on='input')
        meta_pathways = pd.merge(meta_input, self.translation, on='input')
        self.pathways = pd.merge(prot_pathways, meta_pathways, on='translation')
        # self.pathways.to_csv(self.output_dir / 'common_pathways_hmdb.tsv', sep='\t', index=False)

    def extract_expression_data(self):
        prot_id = self.pathways['input_x'].drop_duplicates()
        meta_id = self.pathways['input_y'].drop_duplicates()
        df1_result = self.df1_processed[self.df1_processed['Features'].isin(prot_id)]
        df2_result = self.df2_processed[self.df2_processed['Features'].isin(meta_id)]
        result = pd.concat([df1_result, df2_result], axis=0, ignore_index=True)
        self.expression_matrix = result
        # result.to_csv(self.output_dir / 'prot-meta.tsv', sep='\t', index=False)

    def merge_groups(self):
        group = pd.merge(self.group1, self.group2, on='sample')[['sample', 'group_x']]
        group = group.rename(columns={'group_x': 'group'})
        self.group_matrix = group
        # group.to_csv(self.output_dir / 'prot-meta_group_new.tsv', sep='\t', index=False)

    def run_all(self):
        self.align_samples()
        self.filter_common_pathways()
        self.extract_expression_data()
        self.merge_groups()

    def get_results(self):
        return self.expression_matrix, self.group_matrix
