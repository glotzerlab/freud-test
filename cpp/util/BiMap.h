#ifndef BIMAP_H
#define BIMAP_H

#include <algorithm>
#include <cstddef> // Needed for offsetof
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

/* BiMap container modelled after Boost::BiMap with templatization.
 *
 * "A BiMap is a data structure that represents bidirectional relations between
 * elements of two collections. The container is designed to work as two opposed
 * STL maps. A BiMap between a collection X and a collection Y can be viewed as a
 * map from X to Y or as a map from Y to X. Additionally, the BiMap can also be
 * viewed as a set of relations between X and Y."
 *
 * Public Functions:
 *     BiMap()
 *     ~BiMap()
 *     BiMap(const BiMap&)
 *     BiMap& operator=(const BiMap&)
 *     BiMap& operator=(BiMap&&)
 *
 *     void emplace(Value_A&&, Value_B&&)
 *     void insert (const std::pair<Value_A, Value_B>&)
 *
 *     Left-Side (std::map<Value_A, Value_B>)
 *         const Value_B& at        (const Value_A&)
 *               Value_B& operator[](const Value_A&)
 *               Iterator find      (const Value_A&)
 *               int      count     (const Value_A&)
 *               bool     has       (const Value_A&)
 *               void     erase     (const Value_A&)
 *
 *    Right-Side (std::map<Value_B, Value_A>)
 *         const Value_A& at        (const Value_B&)
 *               Value_A& operator[](const Value_B&)
 *               Iterator find      (const Value_B&)
 *               int      count     (const Value_B&)
 *               bool     has       (const Value_B&)
 *               void     erase     (const Value_B&)
 *
 * Usage:
 *     BiMap<int, string> bm;
 *     bm.emplace(1, "hello");
 *     auto new_pair = make_pair(2, "world");
 *     bm.insert(new_pair);
 *     bm.emplace(3, "how");
 *     bm.emplace(4, "are");
 *     bm.left[5] = "you";
 *
 *     for (int i = 1; i < 6; ++i){
 *         std::cout << bm.left[i] << std::endl;
 *     }
 *
 *     std::cout << bm.right["hello"] << " " << bm.right["world"];
 *
 * Prints:
 *     hello
 *     world
 *     how
 *     are
 *     you
 *     1 2
 */

template<typename T, typename U> class BiMap
{
private:
    template<typename I> struct Comp
    {
        bool operator()(const I* A, const I* B) const
        {
            return *A < *B;
        }
    };

public:
    using Pair = std::pair<T, U>;
    using Container_t = std::vector<Pair*>;
    template<typename I> using Pointer_Set_t = std::set<const I*, Comp<I>>;
    using iterator = typename Container_t::iterator;
    using const_iterator = typename Container_t::const_iterator;
    using reverse_iterator = typename Container_t::reverse_iterator;
    using const_reverse_iterator = typename Container_t::const_reverse_iterator;

private:
    Container_t container;
    Pointer_Set_t<T> set_A;
    Pointer_Set_t<U> set_B;

    template<typename I> Pair* getPairVal(const I* Ptr_in) const
    {
        return const_cast<Pair*>(reinterpret_cast<const Pair*>(Ptr_in));
    }

public:
    BiMap() = default;
    ~BiMap()
    {
        for (size_t i = 0; i < container.size(); ++i)
        {
            delete container[i];
        }
    }

    BiMap(const BiMap& other)
    {
        for (size_t i = 0; i < other.container.size(); ++i)
        {
            this->insert(*(other.container[i]));
        }
    }

    BiMap& operator=(const BiMap& rhs)
    {
        BiMap clone(rhs);
        std::swap(container, clone.container);
        std::swap(set_A, clone.set_A);
        std::swap(set_B, clone.set_B);
        return *this;
    }

    BiMap& operator=(BiMap&& rhs) noexcept
    {
        std::swap(container, rhs.container);
        std::swap(set_A, rhs.set_A);
        std::swap(set_B, rhs.set_B);
        return *this;
    }

    // Return a std::map equivalent to this object.
    std::map<T, U> asMap()
    {
        std::map<T, U> ret;
        for (auto it = begin(); it != end(); ++it)
        {
            ret[(*it)->first] = (*it)->second;
        }
        return ret;
    }

    template<typename I, typename J> bool emplace(I&& Arg1_in, J&& Arg2_in)
    {
        auto pair = new Pair(std::forward<I>(Arg1_in), std::forward<J>(Arg2_in));
        if (set_A.count(&(pair->first)) != 0 || set_B.count(&(pair->second)) != 0)
        {
            delete pair;
            return false;
        }
        set_A.emplace(&(pair->first));
        set_B.emplace(&(pair->second));
        container.emplace_back(std::move(pair));
        return true;
    }

    void insert(const Pair& Pair_in)
    {
        this->emplace(Pair_in.first, Pair_in.second);
    }

    class left
    {
        friend BiMap;

    private:
        BiMap<T, U>& b() const
        {
            BiMap<T, U> t;
            return *reinterpret_cast<BiMap*>(reinterpret_cast<char*>((void*) this)
                                             - (((size_t)(&(&t)->left) - ((size_t) &t))));
        }

        Pair* getPairPtr(const T* Item_in) const
        {
            return b().getPairVal(Item_in);
        }

        U& getVal(const T* Item_in) const
        {
            return getPairPtr(Item_in)->second;
        }

    public:
        const U& at(const T& Key_in) const
        {
            const auto& itr(this->b().set_A.find(&Key_in));
            if (itr == std::end(this->b().set_A))
            {
                throw std::out_of_range {"Key not found"};
            }
            return getVal(*itr);
        }

        U& operator[](const T& Key_in) const
        {
            if (!this->has(Key_in))
            {
                // Add a new item, initializing U to the default
                // then return a reference to U
                b().emplace(Key_in, U());
            }
            return getVal(*(this->b().set_A.find(&Key_in)));
        }

        auto find(const T& Key_in) const -> decltype(this->b().set_A.find(&Key_in))
        {
            return this->b().set_A.find(&Key_in);
        }

        int count(const T& Key_in) const
        {
            return this->b().set_A.count(&Key_in);
        }

        bool has(const T& Key_in) const
        {
            return find(Key_in) != std::end(this->b().set_A);
        }

        void erase(const T& Key_in)
        {
            const auto& pairPtr(getPairPtr(&Key_in));
            this->b().set_A.erase(&(pairPtr->first));
            this->b().set_B.erase(&(pairPtr->second));
            b().container.erase(
                std::remove_if(this->b().container.begin(), this->b().container.end(),
                               [&pairPtr](const std::pair<T, U>* i) { return *i == pairPtr; }),
                this->b().container.end());
        }
    } left;

    friend class left;

    class right
    {
        friend BiMap;

    private:
        BiMap<T, U>& b() const
        {
            BiMap<T, U> t;
            return *reinterpret_cast<BiMap*>(reinterpret_cast<char*>((void*) this)
                                             - (((size_t)(&(&t)->right) - ((size_t) &t))));
        }

        const char* getPairPtr_B(const U* Item_in) const
        {
            return reinterpret_cast<const char*>(Item_in) - offsetof(Pair, second);
        }

        Pair* getPairPtr(const U* Item_in) const
        {
            return b().getPairVal(getPairPtr_B(Item_in));
        }

        T& getVal(const U* Item_in) const
        {
            return getPairPtr(Item_in)->first;
        }

    public:
        const T& at(const U& Key_in) const
        {
            const auto& itr(this->b().set_B.find(&Key_in));
            if (itr == std::end(this->b().set_B))
            {
                throw std::out_of_range {"Key not found"};
            }
            return getVal(*itr);
        }

        T& operator[](const U& Key_in)
        {
            if (!this->has(Key_in))
            {
                // Add a new item, initializing T to the default
                // then return a reference to T
                b().emplace(T(), Key_in);
            }
            return getVal(*(this->b().set_B.find(&Key_in)));
        }

        auto find(const U& Key_in) const -> decltype(this->b().set_B.find(&Key_in))
        {
            return this->b().set_B.find(&Key_in);
        }

        int count(const U& Key_in) const
        {
            return this->b().set_B.count(&Key_in);
        }

        bool has(const U& Key_in) const
        {
            return find(Key_in) != std::end(this->b().set_B);
        }

        void erase(const U& Key_in)
        {
            const auto& pairPtr(getPairPtr(&Key_in));
            this->b().set_A.erase(&(pairPtr->first));
            this->b().set_B.erase(&(pairPtr->second));
            b().container.erase(
                std::remove_if(this->b().container.begin(), this->b().container.end(),
                               [&pairPtr](const std::pair<T, U>* i) { return *i == pairPtr; }),
                this->b().container.end());
        }
    } right;

    friend class right;

    void clear()
    {
        container.clear();
        set_A.clear();
        set_B.clear();
    }

    bool empty() const
    {
        return container.empty();
    }

    size_t size() const
    {
        return container.size();
    }

    auto begin() -> decltype(container.begin())
    {
        return container.begin();
    }

    auto end() -> decltype(container.end())
    {
        return container.end();
    }

    auto cbegin() -> decltype(container.cbegin())
    {
        return container.cbegin();
    }

    auto cend() -> decltype(container.cend())
    {
        return container.cend();
    }

    auto rbegin() -> decltype(container.rbegin())
    {
        return container.rbegin();
    }

    auto rend() -> decltype(container.rend())
    {
        return container.rend();
    }

    auto crbegin() -> decltype(container.crbegin())
    {
        return container.crbegin();
    }

    auto crend() -> decltype(container.crend())
    {
        return container.crend();
    }
};

#endif // BIMAP_H
