#sum array function
def sum_array(arr):
    sum = 0
    for i in arr:
        sum += i
    return sum

#main function
def main():
    arr = [1, 2, 3, 4, 5]
    print("Array is: ")
    print(arr)
    print("Sum of array is: ")
    print(sum_array(arr))

if __name__ == "__main__":
    main()
    
