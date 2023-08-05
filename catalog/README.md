# Reviewer Catalog 
See `ReviewerCatalog.ipynb` to regenerate or contribute to the Reviewer Catalog
## Reviewer Catalog Table
`<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>Description</th>
      <th>url</th>
    </tr>
    <tr>
      <th>Repo</th>
      <th>Type</th>
      <th>Name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="7" valign="top">getzlab/AnnoMate</th>
      <th>Repository</th>
      <th>getzlab/AnnoMate</th>
      <td>Review anything (purities, mutations, etc) within a jupyter notebook with plotly dash and jupyter widgets</td>
      <td><a href="https://github.com/getzlab/AnnoMate" target="_blank">https://github.com/getzlab/AnnoMate</a></td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">AppComponent</th>
      <th>CNVPlotComponent</th>
      <td>CNVPlotComponent.py module

Interactive CNV Plot with mutation multiplicity scatterplot

Mutation scatter interactive with mutation table</td>
      <td><a href="https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/AppComponents/CNVPlotComponent.py" target="_blank">https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/AppComponents/CNVPlotComponent.py</a></td>
    </tr>
    <tr>
      <th>DataTableComponents</th>
      <td>DataTableComponents module contains methods to generate components for displaying table information</td>
      <td><a href="https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/AppComponents/DataTableComponents.py" target="_blank">https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/AppComponents/DataTableComponents.py</a></td>
    </tr>
    <tr>
      <th>MutationTableComponent</th>
      <td>MutationTableComponent.py module

Interactive Mutation Table with column selection, sorting, selecting, and filtering</td>
      <td><a href="https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/AppComponents/MutationTableComponent.py" target="_blank">https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/AppComponents/MutationTableComponent.py</a></td>
    </tr>
    <tr>
      <th>PhylogicComponents</th>
      <td>PhylogicComponents.py module

Phylogic CCF Plot and Trees implemented in the PatientReviewer and PhylogicReviewer

Phylogic PMF Plot implemented in the PhylogicReviewer</td>
      <td><a href="https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/AppComponents/PhylogicComponents.py" target="_blank">https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/AppComponents/PhylogicComponents.py</a></td>
    </tr>
    <tr>
      <th>utils</th>
      <td></td>
      <td><a href="https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/AppComponents/utils.py" target="_blank">https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/AppComponents/utils.py</a></td>
    </tr>
    <tr>
      <th>Reviewer</th>
      <th>ExampleReviewer</th>
      <td>Example Reviewer Description
A basic reviewer for the AnnoMate tutorial.
Uses simulated data from simulated_data directory</td>
      <td><a href="https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/Reviewers/ExampleReviewer.py" target="_blank">https://github.com/getzlab/AnnoMate/blob/master/AnnoMate/Reviewers/ExampleReviewer.py</a></td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">getzlab/PurityReviewer</th>
      <th>Repository</th>
      <th>getzlab/PurityReviewer</th>
      <td>Suite of purity reviewers and review components</td>
      <td><a href="https://github.com/getzlab/PurityReviewer" target="_blank">https://github.com/getzlab/PurityReviewer</a></td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">AppComponent</th>
      <th>AbsoluteCustomSolutionComponent</th>
      <td>Displays a allelic copy ratio profile with the option to set the 0 and 1 line (via slider or input value) corresponding to an integer assignment to copy number peaks. Automatically calculates the purity given the corresponding solution.</td>
      <td><a href="https://github.com/getzlab/PurityReviewer/blob/master/PurityReviewer/AppComponents/AbsoluteCustomSolutionComponent.py" target="_blank">https://github.com/getzlab/PurityReviewer/blob/master/PurityReviewer/AppComponents/AbsoluteCustomSolutionComponent.py</a></td>
    </tr>
    <tr>
      <th>AbsoluteSolutionsReportComponent</th>
      <td>Displays a allelic copy ratio profile and a table of solutions from ABSOLUTE (Carter, 2014). The rows of the table can be selected, and the copy number profile plot will be updated with the corresponding ABSOLUTE "comb" solution.</td>
      <td><a href="https://github.com/getzlab/PurityReviewer/blob/master/PurityReviewer/AppComponents/AbsoluteSolutionsReportComponent.py" target="_blank">https://github.com/getzlab/PurityReviewer/blob/master/PurityReviewer/AppComponents/AbsoluteSolutionsReportComponent.py</a></td>
    </tr>
    <tr>
      <th>utils</th>
      <td></td>
      <td><a href="https://github.com/getzlab/PurityReviewer/blob/master/PurityReviewer/AppComponents/utils.py" target="_blank">https://github.com/getzlab/PurityReviewer/blob/master/PurityReviewer/AppComponents/utils.py</a></td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Reviewer</th>
      <th>ManualPurityReviewer</th>
      <td>A reviewer dashboard that displays generic sample data and a allelic copy ratio profile for a given sample. The allelic copy ratio profile includes sliders and inputs to manually set the 0 and 1 line corresponding to the integer assignment of the genomic segments and automatically calculates a purity.</td>
      <td><a href="https://github.com/getzlab/PurityReviewer/blob/master/PurityReviewer/Reviewers/ManualPurityReviewer.py" target="_blank">https://github.com/getzlab/PurityReviewer/blob/master/PurityReviewer/Reviewers/ManualPurityReviewer.py</a></td>
    </tr>
    <tr>
      <th>MatchedPurityReviewer</th>
      <td>A reviewer dashboard that displays generic sample data and a allelic copy ratio profile for a given sample. The allelic copy ratio profile is linked to a table with the solutions from ABSOLUTE (Carter, 2014), where you can select a row and the corresponding ABSOLUTE "comb" solution will be plotted over the allelic copy ratio plot.</td>
      <td><a href="https://github.com/getzlab/PurityReviewer/blob/master/PurityReviewer/Reviewers/MatchedPurityReviewer.py" target="_blank">https://github.com/getzlab/PurityReviewer/blob/master/PurityReviewer/Reviewers/MatchedPurityReviewer.py</a></td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">getzlab/PatientReviewer</th>
      <th>Repository</th>
      <th>getzlab/PatientReviewer</th>
      <td>Explore integrated data on the patient level interactively in a Dash App, powered by JupyterReviewer</td>
      <td><a href="https://github.com/getzlab/PatientReviewer" target="_blank">https://github.com/getzlab/PatientReviewer</a></td>
    </tr>
    <tr>
      <th>Reviewer</th>
      <th>PatientReviewer</th>
      <td>PatientReviewer.py module

Interactive dashboard for reviewing and annotating data on a patient-by-patient basis
Includes app layout and callback functionality

Run by the user with a Jupyter Notebook: UserPatientReviewer.ipynb</td>
      <td><a href="https://github.com/getzlab/PatientReviewer/blob/master/PatientReviewer/Reviewers/PatientReviewer.py" target="_blank">https://github.com/getzlab/PatientReviewer/blob/master/PatientReviewer/Reviewers/PatientReviewer.py</a></td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">getzlab/MutationReviewer</th>
      <th>Repository</th>
      <th>getzlab/MutationReviewer</th>
      <td></td>
      <td><a href="https://github.com/getzlab/MutationReviewer" target="_blank">https://github.com/getzlab/MutationReviewer</a></td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">AppComponent</th>
      <th>BamTableComponent</th>
      <td>Table of bam files. Each row corresponds to a different bam file, and includes a custom field to reference by sample/patient or other feature. Rows are selectable.</td>
      <td><a href="https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/AppComponents/BamTableComponent.py" target="_blank">https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/AppComponents/BamTableComponent.py</a></td>
    </tr>
    <tr>
      <th>IGVJSComponent</th>
      <td>Displays an internal IGV session inside the dashboard. Takes a list of bams to load and a genomic coordinate to go to.</td>
      <td><a href="https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/AppComponents/IGVJSComponent.py" target="_blank">https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/AppComponents/IGVJSComponent.py</a></td>
    </tr>
    <tr>
      <th>IGVLocalComponent</th>
      <td>Connects with the local IGV app, outside of the dashboard. Takes a list of bams to load and a genomic coordinate to go to.</td>
      <td><a href="https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/AppComponents/IGVLocalComponent.py" target="_blank">https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/AppComponents/IGVLocalComponent.py</a></td>
    </tr>
    <tr>
      <th>MutationTableComponent</th>
      <td>Table of mutations. Mutations can be grouped by different features, and all mutations that are within a group are displayed (ie by patient (tumors and normal) or exclusively by sample).</td>
      <td><a href="https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/AppComponents/MutationTableComponent.py" target="_blank">https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/AppComponents/MutationTableComponent.py</a></td>
    </tr>
    <tr>
      <th>utils</th>
      <td></td>
      <td><a href="https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/AppComponents/utils.py" target="_blank">https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/AppComponents/utils.py</a></td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Reviewer</th>
      <th>DeprecatedMutationReviewer</th>
      <td>Deprecated reviewer. Please see GeneralMutationReviewer.</td>
      <td><a href="https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/Reviewers/DeprecatedMutationReviewer.py" target="_blank">https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/Reviewers/DeprecatedMutationReviewer.py</a></td>
    </tr>
    <tr>
      <th>GeneralMutationReviewer</th>
      <td>A general reviewer for reviewing mutations with IGV. Includes default annotations corresponding to Barnell, 2019 Standard operating procedure for reviewing mutations. Iterates through mutations and automatically loads bams and goes to corresponding coordinates for the mutation in IGV (either local or inside the dashboard itself).</td>
      <td><a href="https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/Reviewers/GeneralMutationReviewer.py" target="_blank">https://github.com/getzlab/MutationReviewer/blob/master/MutationReviewer/Reviewers/GeneralMutationReviewer.py</a></td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">getzlab/PhylogicReviewer</th>
      <th>Repository</th>
      <th>getzlab/PhylogicReviewer</th>
      <td>Interactive app to review Phylogic solutions and data.</td>
      <td><a href="https://github.com/getzlab/PhylogicReviewer" target="_blank">https://github.com/getzlab/PhylogicReviewer</a></td>
    </tr>
    <tr>
      <th>Reviewer</th>
      <th>PhylogicReviewer</th>
      <td></td>
      <td><a href="https://github.com/getzlab/PhylogicReviewer/blob/master/PhylogicReviewer/Reviewers/PhylogicReviewer.py" target="_blank">https://github.com/getzlab/PhylogicReviewer/blob/master/PhylogicReviewer/Reviewers/PhylogicReviewer.py</a></td>
    </tr>
  </tbody>
</table>`