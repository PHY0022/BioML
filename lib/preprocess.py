def GetData(df, l_size, r_size = None):
    if r_size is None:
        r_size = l_size

    sequences = df['Sequence']

    count_pos = 0
    count_neg = 0
    if not ('Positive_Data' in df.columns):  df.insert(len(df.columns), 'Positive_Data', list)
    if not ('Negative_Data' in df.columns): df.insert(len(df.columns), 'Negative_Data', list)
    if not ('Negative_Positions' in df.columns): df.insert(len(df.columns), 'Negative_Positions', list)

    for idx, sequence in enumerate(sequences):
        pos_data = []
        neg_data = []
        pos_positions = df['Positive_Positions'][idx]
        neg_positions = []

        for pos in range(len(sequence)):
            if sequence[pos] == 'K':
                pos += 1 # number in positions is 1-based

                ## Positive data
                if pos in pos_positions:
                    start = pos - l_size - 1
                    end = pos + r_size

                    ### Out of bounds
                    if start < 0 or end > len(sequence):
                        continue

                    pos_data.append(sequence[start:end])
                    count_pos += 1

                ## Negative data
                else:
                    start = pos - l_size - 1
                    end = pos + r_size

                    ### Out of bounds
                    if start < 0 or end > len(sequence):
                        continue

                    ### Overlapping with positive data
                    l_bound = max(0, start - l_size - 1)
                    r_bound = min(len(sequence), end + r_size + 1)
                    if any([p_pos >= l_bound and p_pos <= r_bound or p_pos >= start and p_pos <= end for p_pos in pos_positions]):
                        continue

                    neg_data.append(sequence[start:end])
                    neg_positions.append(pos)
                    count_neg += 1

        df.at[idx, 'Positive_Data'] = pos_data
        df.at[idx, 'Negative_Data'] = neg_data
        df.at[idx, 'Negative_Positions'] = neg_positions


    print(f'Total Positive data: {count_pos}')
    print(f'Total Negative data: {count_neg}')

    return df