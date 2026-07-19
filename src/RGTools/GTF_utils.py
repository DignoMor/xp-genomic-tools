# only works for GENCODE now

import os
import pandas as pd

from .exceptions import RGToolsInternalException, GTFHandleFilterException, GTFRecordNoFeatureException

class GTFRecord:
    def __init__(self, chr_name:str, record_source:str, feature_type:str, start_loc:int, 
                 end_loc:int, score:str, strand:str, phase:str, add_info:dict):
        self.__general_info_dict = dict()
        
        self.__general_info_dict["chr_name"] = chr_name
        self.__general_info_dict["record_source"] = record_source
        self.__general_info_dict["feature_type"] = feature_type
        self.__general_info_dict["start_loc"] = int(start_loc)
        self.__general_info_dict["end_loc"] = int(end_loc)
        self.__general_info_dict["score"] = score
        self.__general_info_dict["strand"] = strand
        self.__general_info_dict["phase"] = phase

        self.__add_info_dict = add_info
    
        # sanity check
        # TODO: add more
        if not self.__general_info_dict["strand"] in ("+", "-"):
            raise RGToolsInternalException("GTF input error. (invalid strandness value: {})".format(self.__general_info_dict["strand"]))
    
    def _print_all_info(self):
        print(self.__general_info_dict)
        print(self.__add_info_dict)

    def search_general_info(self, query):
        if query in self.__general_info_dict.keys():
            return self.__general_info_dict[query]
        else:
            raise GTFRecordNoFeatureException("No feature in general info: {}".format(query))

    def search_add_info(self, query):
        if query in self.__add_info_dict.keys():
            return self.__add_info_dict[query]
        else:
            raise GTFRecordNoFeatureException("No feature in additional info: {}".format(query))
    
class GTFHandle:
    def __init__(self, gtf_path, filter=lambda x: True):
        self.__record_filter = filter
        self.gtf_path = gtf_path

        self.comments = self.__open_gtf_and_read_comments(self.gtf_path)

    def __open_gtf_and_read_comments(self, gtf_path):
        '''
        Read the gtf file, store and comments and point self.__current_line 
        to the first non-comment record
        '''
        comments=[]
        self.__gtf_file = open(gtf_path, 'r')

        self.__current_line= self.__gtf_file.readline()
        while self.__current_line[0] == "#":
            comments.append(self.__current_line)
            self.__current_line = self.__gtf_file.readline()

        return comments

    def __iter__(self):
        self.__gtf_file.close()
        _ = self.__open_gtf_and_read_comments(self.gtf_path)
        return self
    

    def __next__(self):
        while True: 
            # skip empty lines
            while self.__current_line == "\n":
                self.__current_line = self.__gtf_file.readline()
            
            # Stop iteration if eof
            if not self.__current_line: 
                raise StopIteration

            current_record = self.__parse_line(self.__current_line)
            
            if self._is_record_passed_filter(current_record):
                # start again if not passing the filter
                self.__current_line = self.__gtf_file.readline()
                break

            self.__current_line = self.__gtf_file.readline()

        return current_record

    def __del__(self):
        self.__gtf_file.close()
        
    def __parse_line(self, line_to_parse):
        chr_name, record_source, feature_type, start_loc, end_loc, \
        score, strand, phase, add_info = \
            line_to_parse.split("\t")
        
        add_info_dict = dict()

        for add_info_field in add_info.replace("\n", "").rstrip(";").split(";"):
            field_name = add_info_field.split()[0].strip('"')
            field_value = add_info_field.split()[1].strip('"')
            add_info_dict[field_name] = field_value
        
        current_record = GTFRecord(chr_name=chr_name, 
                                   record_source=record_source, 
                                   feature_type=feature_type, 
                                   start_loc=int(start_loc), 
                                   end_loc=int(end_loc), 
                                   score=score, 
                                   strand=strand, 
                                   phase=phase, 
                                   add_info=add_info_dict,
                                   )

        return current_record
    
    def _is_record_passed_filter(self, record):
        # Return False if to filter
        try: 
            is_passed = self.__record_filter(record)
            return is_passed
        except GTFRecordNoFeatureException as e:
            raise GTFHandleFilterException(e)
        
    def get_comments(self):
        comment_lines = [s[1:].lstrip().replace("\n", "") for s in self.comments]
        return "\n".join(comment_lines)
    
    def filter_by_general_record(self, key, value):
        new_record_filter = lambda r: r.search_general_info(key) == value
        record_filter = lambda r: (new_record_filter(r) & self.__record_filter(r))
        return GTFHandle(self.gtf_path, filter=record_filter)

    def filter_by_add_record(self, key, value):
        new_record_filter = lambda r: r.search_add_info(key) == value
        record_filter = lambda r: (new_record_filter(r) & self.__record_filter(r))
        return GTFHandle(self.gtf_path, filter=record_filter)
    
    def count_total(self):
        count = 0
        for record in self:
            count += 1
        
        return count
    
    def filter_check(self, n_lines=None):
        '''
        Check if filter is valid
        '''
        if not n_lines:
            for _ in self:
                pass
        
        else:
            count = 0
            for _ in self:
                count += 1
                
                if count > n_lines:
                    break
    