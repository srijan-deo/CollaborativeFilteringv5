SELECT f.mbr_lic_type, f.mbr_state, f.mbr_nbr, f.mbr_email
FROM `cprtpr-dataplatform-sp1`.usviews.v_us_member_fact f
left join `cprtpr-dataplatform-sp1`.usviews.v_us_bids_fact b
ON f.mbr_nbr = b.buyer_nbr
and b.auc_dt between DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) and CURRENT_DATE()
WHERE mbr_mbrshp_type_cd IN ('BASIC', 'PREMIER')
  AND mbr_status = 'A'
  AND mbr_site_status_cd = 'A'
 and bid_id is null
 and mbr_country = 'USA'
and mbr_lang_pref = 'en'