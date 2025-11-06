CREATE 
    ALGORITHM = UNDEFINED 
    DEFINER = `cwreporting`@`%` 
    SQL SECURITY DEFINER
VIEW `Aurora_Ids_and_keys` AS
    SELECT 
        `vw_distribution_activity`.`vendor_order_key` AS `vendor_order_key`,
        `vw_distribution_activity`.`activity_ref_key` AS `activity_ref_key`,
        `vw_distribution_activity`.`organization_id` AS `organization_id`,
        `vw_distribution_activity`.`organization_name` AS `organization_name`,
        `vw_distribution_activity`.`vendor_id` AS `vendor_id`,
        `vw_distribution_activity`.`vendor_name` AS `vendor_name`,
        `vw_distribution_activity`.`activity_status_key` AS `activity_status_key`,
        `vw_distribution_activity`.`activity_sub_type_description` AS `activity_sub_type_description`,
        `vw_distribution_activity`.`activity_ts` AS `activity_ts`,
        (`vw_distribution_activity`.`total_amount` / 100) AS `total`
    FROM
        `vw_distribution_activity`
    WHERE
        ((`vw_distribution_activity`.`activity_type_id` = 2)
            AND (`vw_distribution_activity`.`updated` IS NULL)
            AND (`vw_distribution_activity`.`organization_id` = 146))