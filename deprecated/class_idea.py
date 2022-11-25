class Collector:
    def __init__(self) -> None:
        # Internal data is probably defined by DPG. The values are not fixed or limited.
        self.collector_topics_internal = collections.deque()

        # External data is probably defined by another model. The values are not fixed or limited.
        self.collector_topics_external = collections.deque()
        self.collector_categories_external = collections.deque()
        self.collector_named_entities_external = collections.deque()

        # The extend of variable values in this data is fixed.
        self.collector_fixed_set_userneeds = collections.deque()
        self.collector_fixed_set_topics = collections.deque()
        self.collector_fixed_set_sensitive_content = collections.deque()  
        
    
    
    def reduce_article(self, fname, orig_dict):
        tmp_dict = orig_dict["enriched_article"]
        new_dict = {}
        new_dict['article_id'] = orig_dict['article_id']
        new_dict['cds_content_id'] = orig_dict['cds_content_id']
        new_dict['brands'] = tmp_dict['brand']
        new_dict['title'] = tmp_dict['title']
        new_dict['text'] = tmp_dict['text_cleaned']
        new_dict['authors'] = tmp_dict['authors']
        new_dict['url'] = tmp_dict.get_str('url')
        new_dict['main_section'] = tmp_dict['section.main_section']
        new_dict['sub_section'] = tmp_dict.get_str('section.sub_section')
        new_dict['num_words'] = tmp_dict.get_int('enrichments.num_words')
        new_dict['num_sentences'] = tmp_dict.get_int('enrichments.num_sentences')
        new_dict['num_chars'] = tmp_dict.get_int('enrichments.raw_character_count')
        new_dict['first_publication_timestamp'] = tmp_dict.get_datetime('first_publication_ts')
        new_dict['categories_generated'] = "|".join(tmp_dict.get_str_list('categories'))
        new_dict['keywords_curated'] = "|".join(tmp_dict.get_str_list('source_keywords'))
        new_dict['brand_safety'] = tmp_dict.get_dict('enrichments.brand_safety')
        new_dict["file_name"] = fname
        
        return benedict(new_dict)

    def reduce_article_topics_external(self, fname, orig_dict):
        def prep(d, nd):
            dnd = {**nd, **d}
            dnd.pop("wikiLink", None)
            return dnd    
        tmp_dict = orig_dict["enriched_article"]
        new_dict = {}
        new_dict['article_id'] = orig_dict['article_id']
        new_dict["file_name"] = fname
        return [prep(d, new_dict) for d in tmp_dict.get_list('enrichments.topics')]

    def reduce_article_topics_internal(self, fname, orig_dict):
        def prep(d, nd):
            dnd = {**nd, **d.pop("media_topic"), "score":d.pop("score")}
            return dnd    
        tmp_dict = orig_dict["enriched_article"]
        new_dict = {}
        new_dict['article_id'] = orig_dict['article_id']
        new_dict["file_name"] = fname
        return [prep(d, new_dict) for d in tmp_dict.get_list('enrichments.semantics.media_topic_inquiry_resultss')]

    def reduce_article_userneeds_fixed_set(self, fname, orig_dict):
        tmp_dict = orig_dict["enriched_article"]
        new_dict = {}
        new_dict['article_id'] = orig_dict['article_id']
        new_dict['userneed'] = tmp_dict.get_dict('enrichments.userneeds.scores')
        new_dict["file_name"] = fname
        return benedict(new_dict).flatten()

    def reduce_article_named_entities(self, fname, orig_dict):
        def prep(d, nd):
            tmp = {}
            all_mentions = [f"{mention.get('begin')}-{mention.get('end')}" for mention in d.get("mentions", [])]
            tmp["mentions"] = ",".join(all_mentions)
            named_entity = d.get('named_entity', {})
            tmp["name"] = named_entity.get('name')
            tmp["type"] = named_entity.get('type')
            tmp["score"] = d.get('score')
            tmp["saliency"] = d.get('saliency')
            dnd = {**tmp, **nd}
            return dnd
        tmp_dict = orig_dict["enriched_article"]
        new_dict = {}
        new_dict['article_id'] = orig_dict['article_id']
        new_dict["file_name"] = fname
        return [prep(d, new_dict) for d in tmp_dict.get_list('enrichments.semantics.named_entity_inquiry_results')]

    def reduce_article_categories_external(fname, orig_dict):
        def prep(d, nd):
            dnd = {**nd, **d}
            dnd.pop("classifierId", None)
            return dnd
            
        tmp_dict = orig_dict["enriched_article"]
        new_dict = {}
        new_dict['article_id'] = orig_dict['article_id']
        new_dict["file_name"] = fname
        return [prep(d, new_dict) for  d in tmp_dict.get_list('enrichments.categories')]

    def reduce_article_topics_fixed_set(fname, orig_dict):
        tmp_dict = orig_dict["enriched_article"]
        new_dict = {}
        new_dict['topic'] = tmp_dict.get_dict('enrichments.ci_topics_v2')
        new_dict['article_id'] = orig_dict['article_id']
        new_dict["file_name"] = fname
        return benedict(new_dict).flatten()

    def reduce_article_sensitive_content_fixed_set(fname, orig_dict):
        tmp_dict = orig_dict["enriched_article"]
        new_dict = {}
        new_dict['sensitive_topic'] = tmp_dict.get_dict('enrichments.garm.categories')
        new_dict['article_id'] = orig_dict['article_id']
        new_dict["file_name"] = fname
        return benedict(new_dict).flatten()