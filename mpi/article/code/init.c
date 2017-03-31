MPI_Request *request = (MPI_Request *)malloc( 5 * sizeof(MPI_Request) );
MPI_Datatype row_type;
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
MPI_Comm_size (MPI_COMM_WORLD, &world_size);
int world_right = (world_rank +1 ) % world_size;
int world_left = (world_rank -1 + world_size) % world_size;
