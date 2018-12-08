import pandas as pd
import ipywidgets as ipw
from IPython.display import HTML

HTML('''<script>
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
</script>
The raw code for this IPython notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

DATA_DIR = 'data/'
SCHEMA = ['statement_id', 'label', 'statement', 'subject', 
          'speaker', 'profession', 'state', 'party', 
          'barely_true', 'false', 'half_true', 
          'mostly_true', 'pants_on_fire', 'context']

# Load the datasets into pandas dataframes
test = pd.read_csv(DATA_DIR + 'test.tsv', delimiter='\t', header=None, names=SCHEMA, index_col=False)
train = pd.read_csv(DATA_DIR + 'train.tsv', delimiter='\t', header=None, names=SCHEMA, index_col=False)
valid = pd.read_csv(DATA_DIR + 'valid.tsv', delimiter='\t', header=None, names=SCHEMA, index_col=False)


# Combine the three dataframes into one
liar = pd.concat([train, test, valid], ignore_index=True)

def clean(s):
    '''
    Replaces upper case letters by lower case ones and removes leading and trailing spaces.
    :param s: str
    :return: s
    '''
    if isinstance(s, str):
        s = s.lower()\
             .strip()
    return s

liar['state'] = liar['state'].apply(clean)

for col_name in liar.columns:
    if not (col_name == 'statement'):
        liar[col_name] = liar[col_name].apply(clean)

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

def lable_proportion(toggle_dots, datapoints_per_dot, label, subject, speaker, profession, state, party, context):
    '''
    Print the proportion of statements which have the properties specified by the inputs.
    :param toggle_dots: bool, datapoints_per_dot: int, label-context: str
    :return: None
    '''
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
        nb_pants_on_fire = round(liarNA[filter_pants_on_fire].shape[0] / datapoints_per_dot)
        nb_false         = round(liarNA[filter_false]        .shape[0] / datapoints_per_dot)
        nb_barely_true   = round(liarNA[filter_barely_true]  .shape[0] / datapoints_per_dot)
        nb_half_true     = round(liarNA[filter_half_true]    .shape[0] / datapoints_per_dot)
        nb_mostly_true   = round(liarNA[filter_mostly_true]  .shape[0] / datapoints_per_dot)
        nb_true          = round(liarNA[filter_true]         .shape[0] / datapoints_per_dot)
    else:
        nb_pants_on_fire = round(liarNA[filter_pants_on_fire & (liarNA.label==label)].shape[0] / datapoints_per_dot)
        nb_false         = round(liarNA[filter_false         & (liarNA.label==label)].shape[0] / datapoints_per_dot)
        nb_barely_true   = round(liarNA[filter_barely_true   & (liarNA.label==label)].shape[0] / datapoints_per_dot)
        nb_half_true     = round(liarNA[filter_half_true     & (liarNA.label==label)].shape[0] / datapoints_per_dot)
        nb_mostly_true   = round(liarNA[filter_mostly_true   & (liarNA.label==label)].shape[0] / datapoints_per_dot)
        nb_true          = round(liarNA[filter_true          & (liarNA.label==label)].shape[0] / datapoints_per_dot)
    
    # Count the number of statements which does not have the properties specified by the inputs
    nb_others = round(nb_tot/datapoints_per_dot - (nb_pants_on_fire + nb_false + nb_barely_true +
                                                   nb_half_true + nb_mostly_true + nb_true))
    
    # Print the legend
    print('\033[1mThere is a total of %s statements, out of which %s satisfy your requirements.\033[0m'%(nb_tot, nb_tot-nb_others*datapoints_per_dot))
    print('\n\033[41m    \033[0m Pants on fire    ' +
            '\033[42m    \033[0m False    '         +
            '\033[43m    \033[0m Barely true    '   +
            '\033[44m    \033[0m Half true    '     +
            '\033[45m    \033[0m Mostly true    '   +
            '\033[46m    \033[0m True    '          +
            '\033[40m    \033[0m Excluded\n')
    
    # Print colored areas proportional to the number of statements satisfying the requirements
    if toggle_dots:# With dots
        print('\033[41m' + html.unescape(nb_pants_on_fire*'&#x25CF') + '\033[0m' +
              '\033[42m' + html.unescape(nb_false*'&#x25CF')         + '\033[0m' +
              '\033[43m' + html.unescape(nb_barely_true*'&#x25CF')   + '\033[0m' +
              '\033[44m' + html.unescape(nb_half_true*'&#x25CF')     + '\033[0m' +
              '\033[45m' + html.unescape(nb_mostly_true*'&#x25CF')   + '\033[0m' +
              '\033[46m' + html.unescape(nb_true*'&#x25CF')          + '\033[0m' +
              '\033[40m' + html.unescape(nb_others*'&#x25CF')        + '\033[0m')
    else:# Without dots
        print('\033[41m' + nb_pants_on_fire*' ' + '\033[0m' +
              '\033[42m' + nb_false*' '         + '\033[0m' +
              '\033[43m' + nb_barely_true*' '   + '\033[0m' +
              '\033[44m' + nb_half_true*' '     + '\033[0m' +
              '\033[45m' + nb_mostly_true*' '   + '\033[0m' +
              '\033[46m' + nb_true*' '          + '\033[0m' +
              '\033[40m' + nb_others*' '        + '\033[0m')

# Create a widget for the function above (lable_proportion)
ipw.interact(lable_proportion,
             toggle_dots = ipw.widgets.ToggleButton(value=False, description='Toggle dots', button_style='success', tooltip='Show/hide the dots representing each statements'),
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