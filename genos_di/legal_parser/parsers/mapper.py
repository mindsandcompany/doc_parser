from collections import defaultdict

from schemas import ParserContent
from commons.loggers import MainLogger

main_logger = MainLogger()

def processor_mapping(article_list:list[ParserContent], addendum_list:list[ParserContent], appendix_list:list[ParserContent]) \
    -> tuple[list[ParserContent],list[ParserContent],list[ParserContent]]:
    """조문, 부칙, 별표 리스트를 받아서 연결 처리를 수행하는 wrapper 함수
    
    Args:
        article_list: 조문 리스트
        addendum_list: 부칙 리스트
        appendix_list: 별표 리스트
        
    Returns:
        article_result: 조문-부칙, 조문-별표가 연결된 조문 리스트
        addendum_result: 부칙-별표가 연결된 부칙 리스트
        appendix_result: 모든 연결이 처리된 별표 리스트

    """
    # 조문 - 부칙 연결
    main_logger.info("[map_article_addenda] 조문 - 부칙 연결 처리")
    mapped_articles, mapped_addendum = map_article_addenda(
        article_list, addendum_list
    )

    # 조문 - 별표 연결
    main_logger.info("[map_article_appendix] 조문 - 별표 연결 처리")
    article_result, mapped_appendices = map_article_appendix(mapped_articles, appendix_list)

    # 부칙 - 별표 연결
    main_logger.info("[map_addendum_appendix] 부칙 - 별표 연결 처리")
    addendum_result, appendix_result = map_addendum_appendix(mapped_addendum, mapped_appendices)

    return article_result, addendum_result, appendix_result

# 조문 <-> 부칙 연결  
def map_article_addenda(article_list: list[ParserContent], addendum_list: list[ParserContent]) \
    -> tuple[list[ParserContent], list[ParserContent]]:
    """공포일자를 기준으로 부칙 메타데이터에 관련 조문 ID를 추가
    조문 메타데이터의 `related_addenda'와 부칙 메타데이터의 `related_articles` 양방향 연결
    """

    article_dict: dict[str, list[ParserContent]] = defaultdict(list)
    for article in article_list:
        article_dict[article.metadata.announce_date].append(article)

    # 이 부칙이 가장 오래된 조문 개정일 이전에 제정되었는지 여부 판단
    oldest_article_date = min(article_dict.keys())

    for index, addendum in enumerate(addendum_list):
        announce_date = addendum.metadata.announce_date
        if index == 0 and oldest_article_date == addendum.metadata.announce_date:
            break
        if announce_date < oldest_article_date:
            addendum.metadata.is_exit = True
  
    # 부칙 데이터를 역순으로 순회하면서 관련 조문 ID 추가
    for addendum in reversed(addendum_list):
        announce_date = addendum.metadata.announce_date

        if announce_date in article_dict:
            for article in article_dict[announce_date]:
                article_id = article.metadata.article_id

                # 부칙에 관련 조문 ID 추가
                addendum.metadata.related_articles = addendum.metadata.related_articles or []
                if article_id not in addendum.metadata.related_articles:
                    addendum.metadata.related_articles.append(article_id)
    
    ## 양방향 연결 
    synchronize_relationships(
        article_list, addendum_list,
        list_a_related_attr="related_addenda", list_b_related_attr="related_articles",
        list_a_id_attr="article_id", list_b_id_attr="addendum_id"
    )

    return article_list, addendum_list

def map_article_appendix(article_list: list[ParserContent], appendix_list: list[ParserContent]) \
    -> tuple[list[ParserContent], list[ParserContent]]:
    """별표의 related_articles와 조문의 related_appendices 양방향 연결
    """
    synchronize_relationships(
        article_list, appendix_list,
        list_a_related_attr="related_appendices", list_b_related_attr="related_articles",
        list_a_id_attr="article_id", list_b_id_attr="appendix_id"
    )

    return article_list, appendix_list

def map_addendum_appendix(addendum_list:list[ParserContent], appendix_list: list[ParserContent])\
    -> tuple[list[ParserContent], list[ParserContent]]:
    """부칙의 related_appendices와 별표의 related_addenda 양방향 연결
    """
    # announce_date 기준으로 appendix를 그룹화
    appendix_dict: dict[str, list[ParserContent]] = defaultdict(list)
    for appendix in appendix_list:
        appendix_dict[appendix.metadata.announce_date].append(appendix)
    
    # 부칙 데이터를 순회하면서 같은 날짜의 별표와 연결
    for addendum in addendum_list:
        announce_date = addendum.metadata.announce_date
        
        if announce_date in appendix_dict:
            for appendix in appendix_dict[announce_date]:
                appendix_id = appendix.metadata.appendix_id
                
                # 부칙에 관련 별표 ID 추가
                addendum.metadata.related_appendices = addendum.metadata.related_appendices or []
                if appendix_id not in addendum.metadata.related_appendices:
                    addendum.metadata.related_appendices.append(appendix_id)
                
                # 별표에 관련 부칙 ID 추가
                addendum_id = addendum.metadata.addendum_id
                appendix.metadata.related_addenda = appendix.metadata.related_addenda or []
                if addendum_id not in appendix.metadata.related_addenda:
                    appendix.metadata.related_addenda.append(addendum_id)
    
    synchronize_relationships(
        addendum_list, appendix_list,
        list_a_related_attr="related_appendices", list_b_related_attr="related_addenda",
        list_a_id_attr="addendum_id", list_b_id_attr="appendix_id"
    )
    return addendum_list, appendix_list


def synchronize_relationships(
    list_a: list[ParserContent], list_b: list[ParserContent], 
    list_a_related_attr: str, list_b_related_attr: str, 
    list_a_id_attr: str, list_b_id_attr: str
):
    """특정 관계 리스트가 존재하는 경우에만, 해당 ID를 반대쪽 리스트에 추가하는 함수.
    
    - list_a의 `list_a_related_attr`(related_*)에 값이 있을 때만 동기화 진행.
    - list_a의 관련 ID를 가진 list_b 항목의 `list_b_related_attr`에 ID를 추가.
    - 반대 방향도 동일하게 적용.

    :param list_a: 관계를 추가할 첫 번째 리스트
    :param list_b: 관계를 추가할 두 번째 리스트
    :param list_a_related_attr: 첫 번째 리스트에서 사용할 related_* 필드명
    :param list_b_related_attr: 두 번째 리스트에서 사용할 related_* 필드명
    :param list_a_id_attr: 첫 번째 리스트에서 ID로 사용할 필드명
    :param list_b_id_attr: 두 번째 리스트에서 ID로 사용할 필드명
    """
    # list_b의 ID를 기준으로 빠르게 찾을 수 있도록 딕셔너리 생성
    list_b_dict = {getattr(item.metadata, list_b_id_attr): item for item in list_b}

    for item_a in list_a:
        related_ids_a = getattr(item_a.metadata, list_a_related_attr) or []
        item_a_id = getattr(item_a.metadata, list_a_id_attr)

        # list_a의 관련 ID 리스트가 비어있으면 아무것도 하지 않음
        if related_ids_a:
            for related_id in related_ids_a:
                item_b = list_b_dict.get(related_id)
                if item_b:
                    related_ids_b = set(getattr(item_b.metadata, list_b_related_attr) or [])

                    # item_a의 ID가 item_b의 related 리스트에 없으면 추가
                    if item_a_id not in related_ids_b:
                        related_ids_b.add(item_a_id)
                        setattr(item_b.metadata, list_b_related_attr, list(related_ids_b))

    # 같은 방식으로 반대 방향도 수행
    list_a_dict = {getattr(item.metadata, list_a_id_attr): item for item in list_a}

    for item_b in list_b:
        related_ids_b = getattr(item_b.metadata, list_b_related_attr) or []
        item_b_id = getattr(item_b.metadata, list_b_id_attr)

        # list_b의 관련 ID 리스트가 비어있으면 아무것도 하지 않음
        if related_ids_b:
            for related_id in related_ids_b:
                item_a = list_a_dict.get(related_id)
                if item_a:
                    related_ids_a = set(getattr(item_a.metadata, list_a_related_attr) or [])

                    # item_b의 ID가 item_a의 related 리스트에 없으면 추가
                    if item_b_id not in related_ids_a:
                        related_ids_a.add(item_b_id)
                        setattr(item_a.metadata, list_a_related_attr, list(related_ids_a))
