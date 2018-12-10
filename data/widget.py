import pandas as pd
import ipywidgets as ipw
import html
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

s = '''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>'''

DATA_DIR = 'data/'

# Load the datasets into pandas dataframes
liar = pd.read_csv(DATA_DIR + 'liar.csv', index_col=0)

# Define function find_top_x
def find_top_x(col_name, x, sorted=False):
    '''
    Returns a sorted (if needed) list of the top x entities with the most occurencies in the column specified by col_name.
    :param col_name: str, x: int, sorted: bool
    :return: top_ten
    '''
    # Get the top x
    top_x = liar[['statement_id', col_name]].groupby(col_name)\
                                            .count()\
                                            .sort_values('statement_id', ascending=False)\
                                            .head(x)\
                                            .index\
                                            .values\
                                            .astype('str')
    
    # Create an array of tupple where the first element is like the second but its dashes are replaced by spaces and each word is capitalized
    top_x = [(s.replace('-', ' ').title(), s) for s in top_x]
    
    # If sorted is True, sort the list by alphabetical order
    if sorted:
        top_x.sort(key=lambda x: x[0])
    return top_x

# Count the total number of statements
nb_tot = liar.shape[0]

# Replace all NaNs by 'NA' to avoid categorizing statements as excluded when they should be included
liarNA = liar.fillna('NA')

# Define function lable_proportion
def lable_proportion(hide_excluded, label, subject, speaker, profession, state, party, context):
    '''
    Print the proportion of statements which have the properties specified by the inputs.
    :param hide_excluded: bool, label-context: str
    :return None
    '''
	# Define the number of datapoints represented by one dot
    datapoints_per_dot = -20
	
    # In case "All <input>" is required, replace the corresponding input by the column it corresponds to in liarNA
    # so that the filter wrt this input contains only "True" values. 
    if subject == 'all_subjects':
        subject = liarNA.subject
    if speaker == 'all_speakers':
        speaker = liarNA.speaker
    if profession == 'all_professions':
        profession = liarNA.profession
    if state == 'all_states':
        state = liarNA.state
    if party == 'all_parties':
        party = liarNA.party
    if context == 'all_contexts':
        context = liarNA.context
    
    # Compute the filters to keep only statements having the properties specified by the inputs
    filter_all_but_labels = (liarNA.subject==subject)       & (liarNA.speaker==speaker) & \
                            (liarNA.profession==profession) & (liarNA.state==state)     & \
                            (liarNA.party==party)           & (liarNA.context==context)
    
    filter_pants_on_fire = (liarNA.label=='pants-fire')  & filter_all_but_labels
    filter_false         = (liarNA.label=='false')       & filter_all_but_labels
    filter_barely_true   = (liarNA.label=='barely-true') & filter_all_but_labels
    filter_half_true     = (liarNA.label=='half-true')   & filter_all_but_labels
    filter_mostly_true   = (liarNA.label=='mostly-true') & filter_all_but_labels
    filter_true          = (liarNA.label=='true')        & filter_all_but_labels
    
    # Apply the filters and count the number of remaining statements for each label.
    if label == 'all_labels':
        nb_pants_on_fire = liarNA[filter_pants_on_fire].shape[0] / datapoints_per_dot
        nb_false         = liarNA[filter_false]        .shape[0] / datapoints_per_dot
        nb_barely_true   = liarNA[filter_barely_true]  .shape[0] / datapoints_per_dot
        nb_half_true     = liarNA[filter_half_true]    .shape[0] / datapoints_per_dot
        nb_mostly_true   = liarNA[filter_mostly_true]  .shape[0] / datapoints_per_dot
        nb_true          = liarNA[filter_true]         .shape[0] / datapoints_per_dot
    else:
        nb_pants_on_fire = liarNA[filter_pants_on_fire & (liarNA.label==label)].shape[0] / datapoints_per_dot
        nb_false         = liarNA[filter_false         & (liarNA.label==label)].shape[0] / datapoints_per_dot
        nb_barely_true   = liarNA[filter_barely_true   & (liarNA.label==label)].shape[0] / datapoints_per_dot
        nb_half_true     = liarNA[filter_half_true     & (liarNA.label==label)].shape[0] / datapoints_per_dot
        nb_mostly_true   = liarNA[filter_mostly_true   & (liarNA.label==label)].shape[0] / datapoints_per_dot
        nb_true          = liarNA[filter_true          & (liarNA.label==label)].shape[0] / datapoints_per_dot
    
    # Count the number of statements which does not have the properties specified by the inputs
    nb_others = nb_tot/datapoints_per_dot - (nb_pants_on_fire + nb_false + nb_barely_true +
                                                   nb_half_true + nb_mostly_true + nb_true)
    
    # Print the legend
    print('\033[1mThere is a total of %s statements, out of which %s satisfy your requirements.\033[0m'%(nb_tot, nb_tot-round(nb_others*datapoints_per_dot)))
    print('\n\033[41m    \033[0m Pants on fire    ' +
            '\033[42m    \033[0m False    '         +
            '\033[43m    \033[0m Barely true    '   +
            '\033[44m    \033[0m Half true    '     +
            '\033[45m    \033[0m Mostly true    '   +
            '\033[46m    \033[0m True    '          +
            '\033[40m    \033[0m Excluded\n')
    
    # Print colored areas proportional to the number of statements satisfying the requirements
    plt.figure(figsize=(20,10))
    plt.fill_between([0,1], 0, nb_pants_on_fire, facecolor=(233/255,91/255,88/255))
    plt.fill_between([0,1], nb_pants_on_fire, nb_pants_on_fire+nb_false, facecolor=(1/255,162/255,80/255))
    plt.fill_between([0,1], nb_pants_on_fire+nb_false, nb_pants_on_fire+nb_false+nb_barely_true, facecolor=(224/255,181/255,42/255))
    plt.fill_between([0,1], nb_pants_on_fire+nb_false+nb_barely_true, nb_pants_on_fire+nb_false+nb_barely_true+nb_half_true, facecolor=(37/255,141/255,246/255))
    plt.fill_between([0,1], nb_pants_on_fire+nb_false+nb_barely_true+nb_half_true, nb_pants_on_fire+nb_false+nb_barely_true+nb_half_true+nb_mostly_true, facecolor=(191/255,85/255,180/255))
    plt.fill_between([0,1], nb_pants_on_fire+nb_false+nb_barely_true+nb_half_true+nb_mostly_true, nb_pants_on_fire+nb_false+nb_barely_true+nb_half_true+nb_mostly_true+nb_true, facecolor=(103/255,194/255,203/255))
    
    if hide_excluded:# Hiding excluded points
        plt.ylim((nb_pants_on_fire+nb_false+nb_barely_true+nb_half_true+nb_mostly_true+nb_true+nb_true,0))
    else:# Showing excluded points
        plt.fill_between([0,1], nb_pants_on_fire+nb_false+nb_barely_true+nb_half_true+nb_mostly_true+nb_true, nb_pants_on_fire+nb_false+nb_barely_true+nb_half_true+nb_mostly_true+nb_true+nb_others, facecolor=(63/255,63/255,63/255))
        plt.ylim((nb_pants_on_fire+nb_false+nb_barely_true+nb_half_true+nb_mostly_true+nb_true+nb_true+nb_others,0))
    
    plt.xlim((0,1))
    plt.axis('off')

# Create a widget for the function above (lable_proportion)
ipw.interact(lable_proportion,
             hide_excluded = ipw.widgets.ToggleButton(value=False, description='Hide \"excluded\"', button_style='success', tooltip='Show/hide the statements which do not satisfy the requirements'),
             datapoints_per_dot = ipw.widgets.IntSlider(value=1., min=1., max=10, description='#statements per dot', style={'description_width': 'initial'}),
             label      = [('All labels', 'all_labels'), ('Pants on fire', 'pants-fire'),
                           ('False', 'false'), ('Barely true', 'barely-true'),
                           ('Half true', 'half-true'), ('Mostly true', 'mostly-true'),
                           ('True', 'true')],
             subject    = [('All subjects',    'all_subjects'   )] + find_top_x('subject',    10, sorted=True),
             speaker    = [('All speakers',    'all_speakers'   )] + find_top_x('speaker',     5, sorted=False),
             profession = [('All professions', 'all_professions')] + find_top_x('profession', 10, sorted=True),
             state      = [('All states',      'all_states'     )] + find_top_x('state',      99, sorted=True),
             party      = [('All parties',     'all_parties'    )] + find_top_x('party',       5, sorted=False),
             context    = [('All contexts',    'all_contexts'   )] + find_top_x('context',    10, sorted=True));
