with filters as
(
select f.mbr_lic_type, f.mbr_state, b.lot_nbr, b.buyer_nbr, f.mbr_email, b.bid_amt, l.inv_dt, l.lot_year, l.lot_make_cd, l.grp_model, l.acv, l.plug_lot_acv, l.repair_cost
from `cprtpr-dataplatform-sp1`.usviews.v_us_bids_fact b
join `cprtpr-dataplatform-sp1`.usviews.v_us_lot_fact l
on b.lot_nbr = l.lot_nbr
join `cprtpr-dataplatform-sp1`.usviews.v_us_member_fact f
on f.mbr_nbr = b.buyer_nbr
where l.inv_dt between DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) and CURRENT_DATE()
and lot_type_cd = 'V' and yard_country_cd = 'USA'
and mbr_country = 'USA' AND mbr_mbrshp_type_cd IN ('BASIC', 'PREMIER')
  AND mbr_site_status_cd = 'A'
AND mbr_mbrshp_status_cd = 'A' AND mbr_lang_pref = 'en'
AND mbr_status = 'A'
),
lots_bid_counts as (
 select lot_nbr, count(distinct buyer_nbr) as total_unique_buyers_on_that_lot
    from filters
    group by 1
 ),
 buyers_bid_counts as (
 select buyer_nbr, count(distinct lot_nbr) as total_unique_lots_bid_by_buyers
 from filters
 group by 1
 )
 select f.mbr_lic_type, f.mbr_state, f.lot_nbr, f.buyer_nbr, f.mbr_email, max(f.bid_amt) as max_bid, f.inv_dt, f.lot_year, f.lot_make_cd, f.grp_model, f.acv, f.plug_lot_acv,
        f.repair_cost, lb.total_unique_buyers_on_that_lot, bb.total_unique_lots_bid_by_buyers
 from filters f
 left join lots_bid_counts as lb on f.lot_nbr = lb.lot_nbr
 left join buyers_bid_counts as bb on f.buyer_nbr = bb.buyer_nbr
 GROUP BY
    f.mbr_lic_type, f.mbr_state, f.lot_nbr, f.buyer_nbr, f.mbr_email, f.inv_dt, f.lot_year, f.lot_make_cd, f.grp_model,
    f.acv, f.plug_lot_acv,f.repair_cost, lb.total_unique_buyers_on_that_lot, bb.total_unique_lots_bid_by_buyers