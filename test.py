from safe_rl.utils.mpi_tools import mpi_fork, proc_id



def foo():
    print('sd')
    def fooo():
        mpi_fork(4)
        print('asas')

    fooo()
    print('ddd')

foo()


