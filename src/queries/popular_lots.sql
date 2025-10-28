WITH cte AS (
  SELECT
    l.buyer_type, f.mbr_state, l.lot_make_cd, l.grp_model,COUNT(*) AS cnt,
    APPROX_QUANTILES(l.acv, 2)[OFFSET(1)] AS median_acv, APPROX_QUANTILES(l.plug_lot_acv, 2)[OFFSET(1)] AS median_plug_lot_acv,
    APPROX_QUANTILES(l.repair_cost, 2)[OFFSET(1)] AS median_repair_cost
  FROM `cprtpr-dataplatform-sp1`.usviews.v_us_member_fact f
  LEFT JOIN `cprtpr-dataplatform-sp1`.usviews.v_us_lot_fact l
    ON f.mbr_nbr = l.buyer_nbr
  WHERE f.mbr_mbrshp_type_cd IN ('BASIC', 'PREMIER')
    AND f.mbr_status = 'A' AND f.mbr_country = 'USA' AND f.mbr_site_status_cd = 'A'
    AND l.inv_dt between DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) and CURRENT_DATE()
    AND l.lot_year >= 2017
    AND l.lot_type_cd = 'V' AND l.yard_country_cd = 'USA'
  and mbr_lang_pref = 'en'
  GROUP BY l.buyer_type, f.mbr_state, l.lot_make_cd, l.grp_model
),

deduped AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY buyer_type, mbr_state, lot_make_cd, grp_model
           ORDER BY cnt DESC
         ) AS model_rank
  FROM cte
),

ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (
           PARTITION BY buyer_type, mbr_state
           ORDER BY cnt DESC
         ) AS rank
  FROM deduped
  WHERE model_rank = 1
)

SELECT *
FROM ranked
WHERE rank <= 6