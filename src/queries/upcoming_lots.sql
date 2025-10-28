SELECT
  lot_nbr, lot_year, lot_make_cd, grp_model, damage_type_desc, repair_cost, acv, plug_lot_acv, auc_dt, COALESCE(proquote_amt, proquote_amt_new) AS proquote_amt
FROM `cprtpr-dataplatform-sp1.usviews.v_us_lot_fact` l
WHERE l.auc_dt BETWEEN DATE_ADD(CURRENT_DATE(), INTERVAL 1 DAY) AND DATE_ADD(CURRENT_DATE(), INTERVAL 7 DAY)
  and loss_type_cd not in ('B','K','F')
    and damage_type_cd not in ('BN','BE','BI','BC')
        and lot_type_cd = 'V' and yard_country_cd = 'USA'
  and offsite_flag = 'N'