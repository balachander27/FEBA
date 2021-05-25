''' 
Authors: Balachander S, Yogesh Chandra Singh Samant, Prahalad Srinivas C G, B Varshin Hariharan
'''
import numpy as np
import pandas as pd
import time
from collections import Counter

def histogram_sampler(data, no_new_data, data_feat, preserve):

    if (data_feat == 'c'):
        start_time = time.time()
        print('Existing data:', len(data))
        print('New data to be produced:', no_new_data)

        '''Function parameters'''

        '''
        X_new = New data for each iteration
        len_X_new = length of the newly generated data
        iter_count = no. of iterations
        data_gen = Augmented data (original data + newly generated data)
        '''
        X_new = []
        len_X_new = len(X_new)
        iter_count = 0
        data_gen = data

        while (len_X_new < 0.7*no_new_data):
            iter_count+=1
            print('-----\niter_count=',iter_count)

            '''
            Histogram: Generating the histogram , choosing the mid-value of the bins, and normalizing frequency
            'fd'- Freedman–Diaconis rule is employed to choose the bin size, as it depends on the spread of the data, without any presumption
            '''

            if(iter_count == 1):
                Y,X_interval=np.histogram(data_gen,bins='doane')
                n_bins = len(Y)

            else:
                Y,X_interval=np.histogram(data_gen,bins=n_bins)

            X = ((X_interval[0:-1] + X_interval[1:])/2) 
            Y = Y/max(Y)

            bin_val = list(np.round(X,8))
            weight = list(Y)
            hist = dict(zip(bin_val,weight))

            for xi in bin_val[0:-1]:

                '''
                Values: choosing the values for undergoing validity check
                '''

                bin_width = ((max(bin_val) - min(bin_val)) / int(len(bin_val)-1))
                xm = xi + (bin_width/2)
                x1 = xi
                y1 = hist[xi]

                res = None
                temp = iter(hist)
                for key in temp:
                    if(key == xi):
                        res = next(temp,None)

                y2 = hist[res]
                ym = ((y1+y2)/2)

                '''
                Validity check: checking if the specified value can be considered
                '''

                '''if(no_new_data <= len(data)):
                    ym = ym*(np.random.rand()<=ym)
                    y1 = y1*(np.random.rand()<=y1)
                '''
                #else:
                ym = ym*(abs(np.random.normal(0,0.50))<=ym)
                y1 = y1*(abs(np.random.normal(0,0.50))<=y1)

                '''
                Appending: appending the valid values
                '''

                if (ym!=0):
                    X_new.append(np.round(xm,8))
                elif (y1!=0):
                    X_new.append(np.round(x1,8))

            '''
            Stopping: bins * 2, length check
            '''

            data_gen = data_gen + X_new
            n_bins = n_bins*2     
            len_X_new+= len(X_new)
            print(len_X_new)
            X_new = []
            print("--- %s seconds ---" % (time.time() - start_time))


        print(len(data_gen)-len(data),no_new_data)

        if(len(data_gen)-len(data) >= no_new_data):
            data_gen = data_gen[:len(data)] + list(np.random.choice(data_gen[len(data):], no_new_data, replace = False))
            print('\nNew data generated:', len(data_gen[len(data):]), '\nNew data:', len(data_gen), '\n')
            #sns.distplot(data_gen)
            if(preserve == False):
                data_gen = data_gen[len(data):]
            return data_gen

        else:
            print('to discrete...', no_new_data - (len(data_gen)-len(data)))
            samples = histogram_sampler(data_gen, no_new_data - (len(data_gen)-len(data)), 'd', preserve = True)
            if(preserve == False):
                samples = samples[len(data):]
            return samples
        
    
    
    elif(data_feat == 'd'):
        X_new=[]
        data_gen=[]
        disc_data= list(set(data))

        for i in disc_data: 
            x=data.count(i)
            X_new.append(round(x*(no_new_data) / len(data)))
        #print(x_new,sum(x_new))

        for j in range(0,len(X_new)):
            for i in range(X_new[j]):
                data_gen.append(disc_data[j])

        if(len(data_gen)==0):
            data_gen = data + data_gen
                    
        if(no_new_data > sum(X_new)):
            data_gen = list(data_gen + list(np.random.choice(data_gen,int(no_new_data-sum(X_new)),replace = False)))    

        data_gen = list(np.random.choice(data_gen,int(no_new_data),replace = False))   
        
        if(preserve == True):
            data_gen = data + data_gen
    
        #sns.distplot(data_gen)     
        print('\nNew data generated:', len(data_gen[len(data):]), '\nNew data:', len(data_gen), '\n') 
        return data_gen
    
    else:
        print('NA')

def label_split(df_class, diff, label_name, cd, label_column, preserve):
    cdi = 0
    df_ = pd.DataFrame(columns=[])
    del df_class[label_column]
    for (columnName, columnData) in df_class.iteritems(): 
        print(columnName)
        feat_type = cd[cdi]
        df_[columnName] = histogram_sampler(list(columnData.values), diff, feat_type, preserve)
        cdi+=1
    df_[label_column] = label_name
    print(df_)

    return df_

def class_balance(data, label_column, cd, augment = False, preserve = True):
    split_list=[]
    #label_column = 'species'
    for label, df_label in data.groupby(label_column):
        split_list.append(df_label)

    maxLength = max(len(x) for x in split_list)

    if(augment == False):
        augmented_list=[]
        for i in range(0,len(split_list)):

            label_name = list(set(split_list[i][label_column]))[0]
            diff = maxLength - len(split_list[i])
            augmented_list.append(label_split(split_list[i], diff, label_name, cd, label_column, preserve))
        finaldf = pd.DataFrame(columns=[])

    elif(type(augment) == dict):
        augmented_list=[]
        for i in range(0,len(split_list)):

            label_name = list(set(split_list[i][label_column]))[0]
            diff = augment[label_name]
            augmented_list.append(label_split(split_list[i], diff, label_name, cd, label_column, preserve))
        finaldf = pd.DataFrame(columns=[])

    elif(type(augment) == int):

        label_count = dict(Counter(list(df[label_column])))
        count_key = list(label_count.keys())
        count_val = list(label_count.values())

        for i in range(0,len(count_val)):
            count_val[i] = round(count_val[i] * augment  /sum(list(label_count.values())))


        while(sum(count_val) != augment):
            if(sum(count_val) > augment):
                rand_indx = int(np.random.rand() * len(count_val))
                if(count_val[rand_indx] > 0):
                    count_val[rand_indx]-= 1

            else:
                rand_indx = int(np.random.rand() * len(count_val))
                count_val[rand_indx]+= 1

        new_count = dict(zip(count_key, count_val))

        augmented_list=[]
        for i in range(0,len(split_list)):

            label_name = list(set(split_list[i][label_column]))[0]
            diff = new_count[label_name]
            augmented_list.append(label_split(split_list[i], diff, label_name, cd, label_column, preserve))
        finaldf = pd.DataFrame(columns=[])


        #finaldf = class_balance(data, label_column, cd, augment = new_count, preserve = preserve)


    for i in range(0,len(split_list)):
        finaldf = pd.concat([finaldf,augmented_list[i]],axis=0)

    return finaldf

