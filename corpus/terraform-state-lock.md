# Terraform State Lock

## Symptoms
`terraform plan` or `terraform apply` fails with `Error acquiring the state
lock` and prints a lock ID, who holds the lock, and when it was created.

## Diagnosis
A lock is normal while another run is in progress. Check the CI pipeline for a
running or recently cancelled Terraform job before doing anything. A lock left
behind by a cancelled job is called a stale lock.

## Remediation
Only after confirming that no Terraform run is active, release a stale lock
with `terraform force-unlock <LOCK_ID>`. Never force-unlock while another apply
is running, because two concurrent applies can corrupt the state file. The
remote state lives in the storage account `tfstateprod` with locking enabled.
