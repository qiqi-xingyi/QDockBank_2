# --*-- conding:utf-8 --*--
# @time:12/3/25 18:45
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:ibm_setup.py

from qiskit_ibm_runtime import QiskitRuntimeService

# ============================================================
# Fill these with your real token and instance for testing
TOKEN = "_WzXCCC92Q86AjZ8AZQsYb-Dew-p_MxgEQb4flrPrtJz"
INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/813b37ffee14414ca81092ab94341434:a6d3c234-40d8-4ddd-afd4-b8ad9b5b4d48::"   # e.g., "ibm-q/open/main"

# ============================================================

def save_runtime_account():
    """
    Save IBM Quantum / Runtime account information locally.
    After running once, QiskitRuntimeService() can load credentials without arguments.
    """
    QiskitRuntimeService.save_account(
        channel="ibm_cloud",   # or "ibm_cloud" if using IBM Cloud
        token=TOKEN,
        instance=INSTANCE,
        overwrite=True
    )
    print("IBM Runtime account information saved successfully.")

if __name__ == "__main__":
    save_runtime_account()
