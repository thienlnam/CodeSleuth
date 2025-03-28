#include <string>
#include <vector>
#include <memory>

// Free function
int add(int a, int b) {
    return a + b;
}

// Function with const and noexcept
inline double multiply(const double& a, const double& b) noexcept {
    return a * b;
}

// Template class
template<typename T>
class Calculator {
public:
    virtual T calculate(const T& a, const T& b) const = 0;
    virtual ~Calculator() = default;
};

// Class with inheritance
class BasicCalculator : public Calculator<int> {
private:
    std::string name;

public:
    explicit BasicCalculator(const std::string& name) : name(name) {}
    
    int calculate(const int& a, const int& b) const override {
        return add(a, b);
    }

    virtual void reset() noexcept {
        // Reset implementation
    }

protected:
    std::string getName() const { return name; }
};

// Class with multiple inheritance
class AdvancedCalculator : protected BasicCalculator, private Calculator<double> {
public:
    AdvancedCalculator() : BasicCalculator("Advanced") {}
    
    double calculate(const double& a, const double& b) const override {
        return multiply(a, b);
    }
    
    static std::unique_ptr<AdvancedCalculator> create() {
        return std::make_unique<AdvancedCalculator>();
    }
}; 