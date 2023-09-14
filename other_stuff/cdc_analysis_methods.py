import numpy as np
import pandas as pd
import string

cdc_cause    = pd.read_csv('data/cdc_by_year_and_icd10_cause_1999_2016.txt', sep = '\t', low_memory = False)

# Clean up data
cdc_cause.rename(columns={'Cause of death':'Cause', 'Cause of death Code' : 'Code'}, inplace=True)
cdc_cause.drop(labels=["Year Code", "Notes", "Crude Rate"], axis=1, inplace=True)

def get_cdc_icd_blocks(icd_block_range_str):
    """
    Returns sorted array of ICD-10 block names (first three chars, e.g. "I05")
    based on occurrences in CDC_cause.
    
        icd_block_range_str:    String containing a description of ICD-10 block name ranges,
                                using a lower and an upper boundary, separated by colons ("-")
    
    Example: "I05-I09" will return ["I05", "I06", ..., "I09"]
    """
    
    def get_icd_block_range(start_block, end_block):
        """
        Returns hypothetical (need not exist in cdc_cause) ICD-10 block range
        """
        
        def get_char_code(char):
            """
            "a" -> "00"
            "b" -> "01"
            ...
            """

            if not len(char) == 1:
                raise ValueError("char must be single letter")

            char_pos = np.where(np.array(list(string.ascii_lowercase)) == char.lower())[0][0]
            char_pos = "{:02d}".format(char_pos)
            return(char_pos)

        def get_codenum(block):
            """
            'A00' -> 0
            'A01' -> 1
            ...
            'A99' -> 99
            'B00' -> 100
            ...
            """
            
            return(int(get_char_code(block[0]) + block[1:3]))

        def get_blocks(codenums):
            """
            E.g., [0, 1, 101] -> ['A00', 'A01', 'B01']
            """
            
            def get_block(codenum):
                char   = list(string.ascii_lowercase)[int(np.floor(codenum/100))].upper()
                digits = "{:02d}".format(int(str(codenum)[-2:]))
                block  = char+digits
                return(block)

            return(list(map(get_block, codenums)))

        start_codenum = get_codenum(start_block)
        end_codenum = get_codenum(end_block)

        codenum_range = np.arange(start_codenum, end_codenum+1)

        block_range = get_blocks(codenum_range)

        return(block_range)
    
    # List with sorted unique Blocks (e.g., "I10") in CDC_cause
    all_cdc_blocks = np.array(cdc_cause['Code'].str.slice(0, 3).unique())
    all_cdc_blocks.sort()

    boundaries = np.array(icd_block_range_str.split("-"))
    if len(boundaries) == 1:
        start_block  = boundaries[0]
        end_block    = start_block
    elif len(boundaries) == 2:
        start_block  = boundaries[0]
        end_block    = boundaries[1]
    
    if(len(start_block) > 3 | len(end_block) > 3):
        print("Length of boundary start or end > 3, warning: information might be truncated")
        start_block  = start_block[:3]
        end_block    = end_block[:3]
    elif(len(start_block) < 3 | len(end_block) < 3):
        raise ValueError("Length of boundary start or end < 3, you might be trying to use this function in a way that hasn't been implemented yet")
    
    hypothetical_icd_block_range = get_icd_block_range(start_block, end_block)
    return(all_cdc_blocks[np.isin(all_cdc_blocks, hypothetical_icd_block_range)])


def get_mortality_by_icd_block_range(icd_block_range_str, years = 'all'):
    """
    Returns number of deaths in cdc_cause
    First output is raw sum, second is sum of #death rel to population (only if only one year)

        icd_block_range_str:    String containing a description of ICD-10 block name ranges,
                                using a lower and an upper boundary, separated by colons ("-").
        year:                   Array of years to filter cdc_cause.

    """
    
    block_range = icd_block_range_str

    if isinstance(years, int):
        years = [years]
    elif years == 'all':
        years = list(cdc_cause['Year'].unique())

    block_list  = get_cdc_icd_blocks(block_range)
    
    cdc_block_and_year_match_idx = np.logical_and(np.isin(cdc_cause['Code'].str.slice(0, 3), block_list),
                                                  np.isin(cdc_cause['Year'], years))
    
    cdc_matches_death_pop = cdc_cause.loc[cdc_block_and_year_match_idx, ['Deaths']]
    return(int(cdc_matches_death_pop.sum(axis=0)))


def get_mortality_by_category(cat_codename_icd_dict, years = "all"):
    """
    Returns number of deaths per category as a df.
    First output is raw sum, second is sum of #death rel to population (only if only one year).

        cat_codename_icd_dict:      dict with category codenames as keys and ICD-10 block range descriptions as values.
        years:                      Array of years to filter cdc_cause.

    """

    categories = []
    mortality = []
    for codename, icd_block_range_str_arr in cat_codename_icd_dict.items():
        cat_sum = 0

        for icd_block_range_str in icd_block_range_str_arr:
            cat_sum += get_mortality_by_icd_block_range(icd_block_range_str, years)

        categories.append(codename)
        mortality.append(cat_sum)
    
    df = pd.DataFrame(np.matrix(mortality).T,
                      index = categories,
                      columns = ['mortality'])
    
    # Now normalize data by dividing by sum of mortality for years in consideration
    mortality_normalized = np.array(mortality)/sum(mortality)
    df['mortality_normalized'] = mortality_normalized
    
    return(df)


def get_raw_rel_mortality_by_cat_and_year_range(cat_dict, year_range):
    """
    Arguments
    ---------
        cat_dict: dict with category names as keys and ICD-10 block ranges as values
        
        year_range: e.g., [2001, 2002, 2003]

    Returns
    -------
    	first: df with raw mortality, second: df with rel mortality
    """
    
    raw_mortality_by_cat_and_year = pd.DataFrame()
    rel_mortality_by_cat_and_year = pd.DataFrame()

    for year in year_range:
        mortality = get_mortality_by_category(cat_dict, year)
        mortality_raw = mortality['raw_mortality']
        mortality_rel = mortality['rel_mortality']
        raw_mortality_by_cat_and_year[year] = mortality_raw
        rel_mortality_by_cat_and_year[year] = mortality_rel

    return(raw_mortality_by_cat_and_year, rel_mortality_by_cat_and_year)
