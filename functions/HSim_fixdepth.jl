using DataFrames
using CSV
using SpecialFunctions
using Plots

####specify inputs (maybe eventually depreciate)
n=7

function hsim_coeffs(t, n, prefact=1/2)

    cheby_even=zeros(Int(n+1))
    cheby_odd=zeros(Int(n+1))

    ###build accoridng to (76, 77) in Martyn, may be able to clean up later
    #then we also assume n is odd
    for l in range(0, Int((n-1)/2))
        cheby_even[2*l+1]=prefact*2*(-1)^l*besselj(2*l, t)
        cheby_odd[2*l+2]=prefact*2*(-1)^l*besselj(2*l+1, t)
    end
    cheby_even[1]=prefact*besselj0(t)
    recip=vcat(reverse(cheby_even[2:end]), vcat(2*cheby_even[1], cheby_even[2:end]))/2
    antirecip=vcat(reverse(cheby_odd[2:end]), vcat(2*cheby_odd[1],cheby_odd[2:end]))/2
    
    return recip, antirecip
end

function LAUR_POLY_BUILD(coeff, n, z)
    polyval=0
    for l in -n:(n)
        polyval=polyval+coeff[l+n+1]*z^l
    end
    return polyval
end


for l in range(4.3, 4.3, 1)
    println(l)
    ceven, codd=hsim_coeffs(l, n)

    x = range(0, π, length=1000)
    z=exp.(im*x)
    y = [exp.(im*l*cos(xterm))/2 for xterm in x] 
    approx=[LAUR_POLY_BUILD(ceven, n, zterm)+im*LAUR_POLY_BUILD(codd, n, zterm) for zterm in z]
    epsi=maximum(abs.(approx.-y))
    println(epsi)

    current_file_path = @__FILE__
    filename="hsim_coeffs_deg_" * string(n) * "_t_" * string(l) * ".csv"
    # filename="hsim_coeffs_eorder_" * string(3) * "_t_" * string(l) * ".csv"
    save_path=replace(replace(current_file_path, "functions" => "csv_files"), "HSim_fixdepth.jl" => filename)
    # print(save_path)
    df= DataFrame(coscoeff=ceven, sincoeff=codd, highest_deg=n, epsilon=epsi, evoltime=l)
    CSV.write(save_path, df)
end


# ceven, codd=hsim_coeffs(t, n)
# x = range(0, π, length=1000)
# y = [exp.(im*t*cos(xterm))/2 for xterm in x] 
# z=exp.(im*x)
# approx=[LAUR_POLY_BUILD(ceven, n, zterm)+im*LAUR_POLY_BUILD(codd, n, zterm) for zterm in z]
# # imagapprox=[ for zterm in z]

# epsi=maximum(abs.(approx.-y))
# print(epsi)
# plot(
#  plot(x, [real(y), real(approx)], label=["cos(t theta)" "approx"]),
#  plot(x, [imag(y), imag(approx)], label=["sin(t theta)" "approx"]),
#  layout=(2, 1)
# )

# current_file_path = @__FILE__
# filename="hsim_coeffs_deg_" * string(n) * "_t_" * string(t) * ".csv"
# save_path=replace(replace(current_file_path, "functions" => "csv_files"), "Hsim.jl" => filename)
# df= DataFrame(coscoeff=ceven, sincoeff=codd, highest_deg=n, epsilon=epsi, evoltime=t)

# if ifsave==true
#     CSV.write(save_path, df)
# end