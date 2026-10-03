# Database Backup and Restore

## Backups
The production PostgreSQL database is backed up nightly at 02:00 UTC with
`pg_dump --format=custom`. Backups are kept for 30 days. The recovery point
objective (RPO) is 24 hours and the recovery time objective (RTO) is 4 hours.

## Restore
Restore into a new database first, never over the production database:
`pg_restore --dbname=<new_db> --jobs=4 <backup_file>`. Verify row counts on key
tables before switching the application to the restored database.

## Restore tests
A restore test is run on the first Monday of every month. A backup that has
never been restored is not considered a working backup.
