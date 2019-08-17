
add_executable(ex_1 ex_1.cpp)
target_link_libraries(ex_1 PRIVATE Common)

add_executable(ex_2 tsallis.cpp)
target_link_libraries(ex_2 PRIVATE dual_annealing)

add_executable(ex_3 ex_3.cpp)
target_link_libraries(ex_3 PRIVATE dual_annealing)

add_executable(ex_4 rastrigin.cpp)
target_link_libraries(ex_4 PRIVATE dual_annealing)
