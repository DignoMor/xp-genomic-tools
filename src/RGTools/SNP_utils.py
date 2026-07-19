
import requests

class EnsemblRestSearch:
    def __init__(self, genome_version='hg38', 
                 species='human', 
                 ):
        '''
        Construct a search object for Ensembl REST API. 
        
        Keyword arguments:
        - genome_version: the genome version to search for. 
          Currently supported versions are:
          - hg38 or GRCh38
          - hg19 or GRCh37
          
          - species: the species to search for. 
          
          Return:
          - An EnsemblRestSearch object
        '''

        if not genome_version in self.genome_version2url_dict.keys():
            raise Exception("Genome version not supported: {}".format(genome_version))

        self.server_url = self.genome_version2url_dict[genome_version]
        self.genome_version = genome_version
        self.species = species

    # API-related methods
    def _get_rsid_info(self, rsid: str):
        '''
        Get all the info of a given rsid, 
        and return a dictionary. This methods 
        returns the unmodified full response 
        from ENSEMBL REST API. The coordinates 
        in this response are 1-based and closed.

        Keyword arguments:
        - rsid: the rsid to search for

        Return:
        - a dictionary containing the info of the rsid
        '''
        ext = "/variation/{}/{}?genotypes=1".format(self.species, 
                                                    rsid, 
                                                    )
        api_url = self.server_url + ext

        headers = { "Content-Type" : "application/json"}

        r = requests.get(api_url, headers=headers)

        if not r.ok:
            r.raise_for_status()

        return r.json()

    def get_rsid_from_location(self, chrom: str, pos: int):
        '''
        Given a chromosome and a position, return a list of the rsids 
        of the SNP at that location. 

        The Only variants with the exact start and end position 
        at the given pos will be returned. (i.e. only SNPs and 
        indels with length 1)

        Keyword arguments:
        - chrom: the chromosome to search for
        - pos: the position to search for. This is 0-based.

        Return:
        - the rsid of the SNP at that location
        '''
        # ENSEMBL REST API uses 1-based position
        pos += 1
        ext = "/overlap/region/{}/{}:{:d}-{:d}?feature=variation".format(self.species,
                                                                         str(chrom).replace("chr", ""), 
                                                                         pos, 
                                                                         pos, 
                                                                         )
        api_url = self.server_url + ext

        headers = { "Content-Type" : "application/json"}

        r = requests.get(api_url, headers=headers)

        if not r.ok:
            r.raise_for_status()
        
        return [d["id"] for d in r.json() if (d["start"] == pos and d["end"] == pos)]

    def get_rsid_snp_simple_info(self, rsid: str):
        '''
        Get the SNP info of a given rsid, 
        and return a dictionary.

        Keyword arguments:
        - rsid: the rsid to search for

        Return:
        - a dictionary containing the following info 
          of the rsid: ["chrom", "start", "end", "bases"].
          The start and end coordinates follows UCSC bed convention 
          (0-based, half-open).
        '''
        info = self._get_rsid_info(rsid)
        for mapping in info["mappings"]:
            if mapping["coord_system"] == "chromosome":
                return {"chrom": "chr"+mapping["seq_region_name"], # UCSC format in this lib
                        "start": mapping["start"]-1, 
                        "end": mapping["end"],
                        "bases": mapping["allele_string"],
                        }
            
        # if not found, raise an exception
        raise Exception("No chromosome coord found in the mapping for rsid: {}".format(rsid))

    # End of API-related methods 

    @property
    def genome_version2url_dict(self):
        '''
        A dictionary mapping genome version to the corresponding
        Ensembl REST API URL.
        '''
        return {
            "hg38": "https://rest.ensembl.org",
            "GRCh38": "https://rest.ensembl.org",
            "hg19": "https://grch37.rest.ensembl.org",
            "GRCh37": "https://grch37.rest.ensembl.org",
        }

    def _is_SNP(self, rsid):
        '''
        Check if the variant corresponding to an rsid is a SNP

        Keyword arguments:
        - rsid: the rsid to search for

        Return:
        - is_snp: True if the variant is a SNP, False otherwise
        - simple_snp_info of the rsid
        '''
        is_snp = True
        simple_snp_info = self.get_rsid_snp_simple_info(rsid)

        # If end - start > 1, it is not a SNP
        is_snp = ((simple_snp_info["end"] - simple_snp_info["start"]) == 1)

        # If there are indels with length > 1, it is not a SNP
        base_list = simple_snp_info["bases"].split("/")
        if len(base_list) < 2:
            is_snp = False

        for base in base_list:
            if len(base) > 1:
                is_snp = False
        
        return is_snp, simple_snp_info

    def prioritize_rsids(self, rsids: "Iterable"):
        '''
        Run a set of filters to find the most SNP-like rsid

        Keyword arguments:
        - rsids: Iterable of rsids to search for

        Return: 
        - The most prioritized rsid and snp info dict. If 
          no SNP is in the list, return None.
        - simple_snp_info of the most prioritized rsid, 
          None if no SNP is in the list.
        '''
        snp_rsid_list = []
        snp_simple_info_list = []
        for rsid in rsids:
            is_snp, simple_snp_info = self._is_SNP(rsid)
            if is_snp:
                snp_rsid_list.append(rsid)
                snp_simple_info_list.append(simple_snp_info)

        if len(snp_rsid_list) == 0:
            return None, None
        elif len(snp_rsid_list) == 1:
            return snp_rsid_list[0], snp_simple_info_list[0]
        else:
            # return the one with most alternate alleles
            allele_count_list = [len(snp_info["bases"].split("/")) for snp_info in snp_simple_info_list]
            ind_max_count = allele_count_list.index(max(allele_count_list))
            return snp_rsid_list[ind_max_count], snp_simple_info_list[ind_max_count]
